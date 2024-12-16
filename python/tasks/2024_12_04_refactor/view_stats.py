import pstats

stats = pstats.Stats('cpu.prof')
stats.sort_stats('cumtime').print_stats(30)
