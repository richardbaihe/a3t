"""Parallel beam search module."""

import logging
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from espnet.nets.beam_search import BeamSearch
from espnet.nets.beam_search import Hypothesis


class BatchHypothesis(NamedTuple):
    """Batchfied/Vectorized hypothesis data type."""

    yseq: torch.Tensor = torch.tensor([])     # (batch, maxlen)
    score: torch.Tensor = torch.tensor([])    # (batch,)
    length: torch.Tensor = torch.tensor([])   # (batch,)
    scores: Dict[str, torch.Tensor] = dict()  # values: (batch,)
    states: Dict[str, Dict] = dict()

    def __len__(self) -> int:
        """Return a batch size."""
        return len(self.length)


class BatchBeamSearch(BeamSearch):
    """Batch beam search implementation."""

    def batchfy(self, hyps: List[Hypothesis]) -> BatchHypothesis:
        """Convert list to batch."""
        if len(hyps) == 0:
            return BatchHypothesis()
        return BatchHypothesis(
            yseq=pad_sequence([h.yseq for h in hyps], batch_first=True, padding_value=self.eos),
            length=torch.tensor([len(h.yseq) for h in hyps], dtype=torch.int64),
            score=torch.tensor([h.score for h in hyps]),
            scores={k: torch.tensor([h.scores[k] for h in hyps]) for k in self.scorers},
            states={k: [h.states[k] for h in hyps] for k in self.scorers}
        )

    def _batch_select(self, hyps: BatchHypothesis, ids: List[int]) -> BatchHypothesis:
        return BatchHypothesis(
            yseq=hyps.yseq[ids],
            score=hyps.score[ids],
            length=hyps.length[ids],
            scores={k: v[ids] for k, v in hyps.scores.items()},
            states={k: [self.scorers[k].select_state(v, i) for i in ids]
                    for k, v in hyps.states.items()},
        )

    def _select(self, hyps: BatchHypothesis, i: int) -> Hypothesis:
        return Hypothesis(
            yseq=hyps.yseq[i, :hyps.length[i]],
            score=hyps.score[i],
            scores={k: v[i] for k, v in hyps.scores.items()},
            states={k: self.scorers[k].select_state(v, i) for k, v in hyps.states.items()},
        )

    def unbatchfy(self, batch_hyps: BatchHypothesis) -> List[Hypothesis]:
        """Revert batch to list."""
        return [
            Hypothesis(
                yseq=batch_hyps.yseq[i][:batch_hyps.length[i]],
                score=batch_hyps.score[i],
                scores={k: batch_hyps.scores[k][i] for k in self.scorers},
                states={k: v.select_state(
                    batch_hyps.states[k], i) for k, v in self.scorers.items()}
            ) for i in range(len(batch_hyps.length))]

    def batch_beam(self, weighted_scores: torch.Tensor, ids: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch-compute topk full token ids and partial token ids.

        Args:
            weighted_scores (torch.Tensor): The weighted sum scores for each tokens.
                Its shape is `(n_beam, self.vocab_size)`.
            ids (torch.Tensor): The partial token ids to compute topk.
                Its shape is `(n_beam, self.pre_beam_size)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                The topk full (prev_hyp, new_token) ids and partial (prev_hyp, new_token) ids.
                Their shapes are all `(self.beam_size,)`

        """
        if not self.do_pre_beam:
            top_ids = weighted_scores.view(-1).topk(self.beam_size)[1]
            # Because of the flatten above, `top_ids` is organized as:
            # [hyp1 * V + token1, hyp2 * V + token2, ..., hypK * V + tokenK],
            # where V is `self.n_vocab` and K is `self.beam_size`
            prev_hyp_ids = top_ids // self.n_vocab
            new_token_ids = top_ids % self.n_vocab
            return prev_hyp_ids, new_token_ids, prev_hyp_ids, new_token_ids

        raise NotImplementedError("batch decoding with PartialScorer is not supported yet.")

    def init_hyp(self, x: torch.Tensor) -> BatchHypothesis:
        """Get an initial hypothesis data.

        Args:
            x (torch.Tensor): The encoder output feature

        Returns:
            Hypothesis: The initial hypothesis.

        """
        init_states = dict()
        init_scores = dict()
        for k, d in self.scorers.items():
            init_states[k] = d.init_state(x)
            init_scores[k] = 0.0
        return self.batchfy(super().init_hyp(x))

    def search(self, running_hyps: BatchHypothesis, x: torch.Tensor) -> BatchHypothesis:
        """Search new tokens for running hypotheses and encoded speech x.

        Args:
            running_hyps (BatchHypothesis): Running hypotheses on beam
            x (torch.Tensor): Encoded speech feature (T, D)

        Returns:
            BatchHypothesis: Best sorted hypotheses

        """
        n_batch = len(running_hyps)

        # batch scoring
        scores, states = self.score_full(running_hyps, x.expand(n_batch, *x.shape))
        if self.do_pre_beam:
            part_ids = torch.topk(scores[self.pre_beam_score_key], self.pre_beam_size, dim=-1)[1]
        else:
            part_ids = torch.arange(self.n_vocab, device=x.device).expand(n_batch, self.n_vocab)
        part_scores, part_states = self.score_partial(running_hyps, part_ids, x)

        # weighted sum scores
        weighted_scores = torch.zeros(n_batch, self.n_vocab, dtype=x.dtype, device=x.device)
        for k in self.full_scorers:
            weighted_scores += self.weights[k] * scores[k]
        for k in self.part_scorers:
            weighted_scores[part_ids] += self.weights[k] * part_scores[k]
        weighted_scores += running_hyps.score.unsqueeze(1)

        # TODO(karita): do not use list. use batch instead
        # update hyps
        best_hyps = []
        prev_hyps = self.unbatchfy(running_hyps)
        for full_prev_hyp_id, full_new_token_id, part_prev_hyp_id, part_new_token_id in zip(
                *self.batch_beam(weighted_scores, part_ids)):
            prev_hyp = prev_hyps[full_prev_hyp_id]
            best_hyps.append(Hypothesis(
                score=weighted_scores[full_prev_hyp_id, full_new_token_id],
                yseq=self.append_token(prev_hyp.yseq, full_new_token_id),
                scores=self.merge_scores(
                    prev_hyp.scores,
                    {k: v[full_prev_hyp_id] for k, v in scores.items()}, full_new_token_id,
                    {k: v[part_prev_hyp_id] for k, v in part_scores.items()}, part_new_token_id),
                states=self.merge_states(
                    {k: self.full_scorers[k].select_state(v, full_prev_hyp_id) for k, v in states.items()},
                    {k: self.part_scorers[k].select_state(v, part_prev_hyp_id) for k, v in part_states.items()},
                    part_new_token_id)
            ))
        return self.batchfy(best_hyps)

    def post_process(self, i: int, maxlen: int, maxlenratio: float,
                     running_hyps: BatchHypothesis, ended_hyps: List[Hypothesis]) -> BatchHypothesis:
        """Perform post-processing of beam search iterations.

        Args:
            i (int): The length of hypothesis tokens.
            maxlen (int): The maximum length of tokens in beam search.
            maxlenratio (int): The maximum length ratio in beam search.
            running_hyps (BatchHypothesis): The running hypotheses in beam search.
            ended_hyps (List[Hypothesis]): The ended hypotheses in beam search.

        Returns:
            BatchHypothesis: The new running hypotheses.

        """
        n_batch, maxlen = running_hyps.yseq.shape
        logging.debug(f'the number of running hypothes: {n_batch}')
        if self.token_list is not None:
            logging.debug("best hypo: " + "".join(
                [self.token_list[x] for x in running_hyps.yseq[0, 1:running_hyps.length[0]]]))
        # add eos in the final loop to avoid that there are no ended hyps
        if i == maxlen - 1:
            logging.info("adding <eos> in the last position in the loop")
            running_hyps.yseq.resize_(n_batch, maxlen + 1)
            running_hyps.yseq[:, -1] = self.eos
            running_hyps.yseq.index_fill_(1, running_hyps.length, self.eos)

        # add ended hypotheses to a final list, and removed them from current hypotheses
        # (this will be a probmlem, number of hyps < beam)
        is_eos = running_hyps.yseq[torch.arange(n_batch), running_hyps.length - 1] == self.eos
        for b in torch.nonzero(is_eos).view(-1):
            hyp = self._select(running_hyps, b)
            ended_hyps.append(hyp)
        remained_ids = torch.nonzero(is_eos == 0).view(-1)
        return self._batch_select(running_hyps, remained_ids)
