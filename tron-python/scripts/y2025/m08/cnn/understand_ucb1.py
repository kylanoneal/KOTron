import numpy as np
import matplotlib.pyplot as plt

        
def main():
        
    # --- Parameters you can tweak ---
    c = 2.0  # exploration_factor
    N_min, N_max = 1, 10_000   # parent visits (node.n_visits). Must be >= 1 (log(1)=0)
    n_min, n_max = 1, 5_000    # child visits + 1 (curr_child_visits). Must be >= 1

    # --- Define the exploration term: c * sqrt(log(N) / n) ---
    def exploration_value(N, n, c):
        N = np.asarray(N, dtype=float)
        n = np.asarray(n, dtype=float)
        # Safe: log(1)=0 gives 0 exploration; N>=1 and n>=1 by construction
        return c * np.sqrt(np.log(N) / n)

    # ===== 1) Heatmap over a grid of N (parent visits) and n (child visits) =====
    # Use log spacing to see behavior across orders of magnitude
    N_vals = np.logspace(np.log10(N_min), np.log10(N_max), 300)
    n_vals = np.logspace(np.log10(n_min), np.log10(n_max), 300)
    n_grid, N_grid = np.meshgrid(n_vals, N_vals, indexing='ij')

    E = exploration_value(N_grid, n_grid, c)


    plt.figure(figsize=(8.5, 6.5))

    pc = plt.pcolormesh(N_vals, n_vals, E, shading='auto')
    plt.xscale('log')
    plt.yscale('log')
    plt.colorbar(pc, label='Exploration value')
    plt.xlabel('Parent visits N')
    plt.ylabel('Child visits n')
    plt.title('UCB1 Exploration Term Heatmap (log-log axes)')
    plt.tight_layout()
    # Use imshow with log axes; provide extent in data coords


    # # ===== 2) Curves: exploration vs child visits for fixed parent visits =====
    fixed_N_list = [10, 100, 1_000, 10_000, 100_000]
    n_curve = np.logspace(np.log10(n_min), np.log10(n_max), 400)

    plt.figure(figsize=(8.5, 6.0))
    for N in fixed_N_list:
        y = exploration_value(N, n_curve, c)
        plt.plot(n_curve, y, label=f'N={N}')
    plt.xscale('log')
    plt.xlabel('Child visits n = curr_child_visits')
    plt.ylabel('Exploration value')
    plt.title('Exploration vs Child Visits (various parent visits N)')
    plt.legend()
    plt.grid(True, which='both', alpha=0.25)
    plt.tight_layout()

    # ===== 3) Curves: exploration vs parent visits for fixed child visits =====
    fixed_n_list = [1, 2, 5, 10, 50, 100, 500, 1000]
    N_curve = np.logspace(np.log10(N_min), np.log10(N_max), 400)

    plt.figure(figsize=(8.5, 6.0))
    for n in fixed_n_list:
        y = exploration_value(N_curve, n, c)
        plt.plot(N_curve, y, label=f'n={n}')
    plt.xscale('log')
    plt.xlabel('Parent visits N = node.n_visits')
    plt.ylabel('Exploration value')
    plt.title('Exploration vs Parent Visits (various child visits n)')
    plt.legend(ncol=2)
    plt.grid(True, which='both', alpha=0.25)
    plt.tight_layout()

    plt.show()

if __name__=="__main__":
    main()