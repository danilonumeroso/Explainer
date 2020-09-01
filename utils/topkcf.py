from config.explainer import Path, Args

class TopKCounterfactuals:
    Leaderboard = None
    K = 5

    @staticmethod
    def init(original, index, k=5):

        TopKCounterfactuals.K = k

        if TopKCounterfactuals.Leaderboard is None:
            TopKCounterfactuals.Leaderboard = {
                'original': original,
                'index': index,
                'counterfacts': [
                    {'smiles': '', 'score': -0.1}
                    for _ in range(k)
                ]
            }

    @staticmethod
    def insert(counterfact):

        Leaderboard = TopKCounterfactuals.Leaderboard
        K = TopKCounterfactuals.K

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

        TopKCounterfactuals._dump()

    @staticmethod
    def _dump():
        import json

        with open(
            Path.counterfacts(
                str(TopKCounterfactuals.Leaderboard['index']) + '.json',
                "Tox21"
            ),
            'w'
        ) as f:
            json.dump(TopKCounterfactuals.Leaderboard, f, indent=2)


class TopKCounterfactualsESOL:
    Leaderboard = None
    K = 5

    @staticmethod
    def init(original, index, k=5):

        TopKCounterfactuals.K = k

        if TopKCounterfactuals.Leaderboard is None:
            TopKCounterfactuals.Leaderboard = {
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

        Leaderboard = TopKCounterfactuals.Leaderboard
        K = TopKCounterfactuals.K

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

        TopKCounterfactuals._dump()

    @staticmethod
    def _dump():
        import json

        with open(
            Path.counterfacts(
                str(TopKCounterfactuals.Leaderboard['index']) + '.json',
                "ESOL"
            ),
            'w'
        ) as f:
            json.dump(TopKCounterfactuals.Leaderboard, f, indent=2)
