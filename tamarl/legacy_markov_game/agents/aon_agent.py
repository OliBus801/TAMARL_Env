from tamarl.rl_models.msa_agent import MSAAgent

class AONAgent(MSAAgent):
    """
    All-or-Nothing Agent.
    Calcule le chemin le plus court en flux libre (Free Flow) au début
    et ne change jamais de route, simulant un scénario sans intelligence adaptative.
    """
    def __init__(self, **kwargs):
        # On hérite de MSA car la logique de suivi de chemin est identique
        super().__init__(**kwargs)
        self.calculated = False

    def end_episode(self, dnl, is_init=False):
        # L'AON ne calcule ses routes qu'une seule fois au début (is_init=True)
        if not self.calculated and is_init:
            super().end_episode(dnl, is_init=True)
            self.calculated = True
        else:
            # On ne fait rien lors des fins d'épisodes suivantes
            pass

    def __repr__(self) -> str:
        return "AONAgent(Shortest Path at Free Flow)"
