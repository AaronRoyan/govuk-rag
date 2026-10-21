# Retrieval results

70 answerable questions, scored at guide-part level. Latency is per query on an Apple M2 CPU.

| Mode | Hit@1 | Hit@3 | Hit@5 | MRR | nDCG@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| `bm25` | 0.486 | 0.814 | 0.900 | 0.657 | 0.695 | 0.7 | 1.1 |
| `dense` | 0.757 | 0.971 | 0.971 | 0.858 | 0.865 | 6.1 | 9.4 |

## Hit@3 by question type

| Mode | lookup | multi-part | numeric | paraphrase | reasoning |
|---|---|---|---|---|---|
| `bm25` | 0.89 | 0.71 | 0.84 | 0.78 | 0.83 |
| `dense` | 0.89 | 1.00 | 1.00 | 1.00 | 0.92 |

## Misses for `dense` (gold not in top 3)

- **q016** What is a MATB1 and why do I need one?  
  gold: `maternity-pay-leave/how-to-claim` got: `agency-workers-your-rights/basic-information-you-should-receive, flexible-working/overview, flexible-working/applying-for-flexible-working`
- **q053** I'm classed as a worker, not an employee. Do I get statutory redundancy pay?  
  gold: `employment-status/worker` got: `redundancy-your-rights/redundancy-pay, redundancy-your-rights/suitable-alternative-employment, employment-status/employee`
