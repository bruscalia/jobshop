import pyomo.environ as pyo
import matplotlib as mpl
import matplotlib.pyplot as plt


class JobShopModel(pyo.ConcreteModel):
    
    cmap = mpl.colormaps["Dark2"]
    colors = cmap.colors
    
    def __init__(self, params, **kwargs):
        super().__init__()
        self.params = params
        self._construct_sets()
        self._construct_params()
    
    @property
    def seq(self):
        return self.params.seq
    
    def _construct_sets(self):
        self.M = pyo.Set(initialize=self.params.machines)
        self.J = pyo.Set(initialize=self.params.jobs)
    
    def _construct_params(self):
        self.p = pyo.Param(self.M, self.J, initialize=self.params.p_times)
        self.V = sum(self.p[key] for key in self.p)
    
    def plot(self, horizontal=True, figsize=[7, 3], dpi=100, colors=None):
        if horizontal:
            self._plot_horizontal(figsize=figsize, dpi=dpi, colors=colors)
        else:
            self._plot_vertical(figsize=figsize, dpi=dpi, colors=colors)

    def _plot_vertical(self, figsize=[7, 3], dpi=100, colors=None):
        
        if colors is None:
            colors = self.colors
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        for i, j in enumerate(self.J):
            machines, starts, spans = self._get_elements(j)
            
            if i >= len(colors):
                i = i % len(colors)
            
            color = colors[i]
            ax.bar(machines, spans, bottom=starts, label=f"Job {j}", color=color)

        ax.set_xticks(self.M)
        ax.set_xlabel("Machine")
        ax.set_ylabel("Time")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1.03))
        fig.tight_layout()
        plt.show()

    def _plot_horizontal(self, figsize=[7, 3], dpi=100, colors=None):
        
        colors = self._get_colors(colors)
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        for i, j in enumerate(self.J):
            machines, starts, spans = self._get_elements(j)
            
            if i >= len(colors):
                i = i % len(colors)
            
            color = colors[i]
            ax.barh(machines, spans, left=starts, label=f"Job {j}", color=color)

        ax.set_yticks(self.M)
        ax.set_xlabel("Time")
        ax.set_ylabel("Machine")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1.03))
        fig.tight_layout()
        plt.show()
    
    def _get_elements(self, j):
        pass
    
    def _get_colors(self, colors):
        if colors is None:
            colors = self.colors
        return colors
