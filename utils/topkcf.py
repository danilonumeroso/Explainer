from config.explainer import Path, Args

class TopKCounterfactualsTox21:
    Leaderboard = None
    K = 5

    @staticmethod
    def init(original, index, k=5):

        TopKCounterfactualsTox21.K = k

        if TopKCounterfactualsTox21.Leaderboard is None:
            TopKCounterfactualsTox21.Leaderboard = {
                'original': original,
                'index': index,
                'counterfacts': [
                    {'smiles': '', 'score': -0.1}
                    for _ in range(k)
                ]
            }

    @staticmethod
    def insert(counterfact):

        Leaderboard = TopKCounterfactualsTox21.Leaderboard
        K = TopKCounterfactualsTox21.K

        if any(
            x['smiles'] == counterfact['smiles']
            for x in Leaderboard['counterfacts']
        ):
            return

        Leaderboard['counterfacts'].extend([counterfact])
        Leaderboard['counterfacts'].sort(
            reverse=True,
            key=lambda x: x['score']
        )
        Leaderboard['counterfacts'] = Leaderboard['counterfacts'][:K]

        TopKCounterfactualsTox21._dump()

    @staticmethod
    def _dump():
        import json

        with open(
            Path.counterfacts(
                str(TopKCounterfactualsTox21.Leaderboard['index']) + '.json',
                "Tox21"
            ),
            'w'
        ) as f:
            json.dump(TopKCounterfactualsTox21.Leaderboard, f, indent=2)


class TopKCounterfactualsESOL:
    Leaderboard = None
    K = 5

    @staticmethod
    def init(original, index, k=5):

        TopKCounterfactualsESOL.K = k

        if TopKCounterfactualsESOL.Leaderboard is None:
            TopKCounterfactualsESOL.Leaderboard = {
                'original': original,
                'index': index,
                'counterfacts': [
                    {
                        'smiles': '',
                        'score': -0.1,
                        'loss': 0,
                        'gain': 0,
                        'sim': 0
                    }
                    for _ in range(k)
                ]
            }

    @staticmethod
    def insert(counterfact):

        Leaderboard = TopKCounterfactualsESOL.Leaderboard
        K = TopKCounterfactualsESOL.K

        if any(
            x['smiles'] == counterfact['smiles']
            for x in Leaderboard['counterfacts']
        ):
            return

        Leaderboard['counterfacts'].extend([counterfact])
        Leaderboard['counterfacts'].sort(
            reverse=True,
            key=lambda x: x['score']
        )
        Leaderboard['counterfacts'] = Leaderboard['counterfacts'][:K]

        TopKCounterfactualsESOL._dump()

    @staticmethod
    def _dump():
        import json
        with open(
            Path.counterfacts(
                str(TopKCounterfactualsESOL.Leaderboard['index']) + '.json',
                "ESOL"
            ),
            'w'
        ) as f:
            json.dump(TopKCounterfactualsESOL.Leaderboard, f, indent=2)
