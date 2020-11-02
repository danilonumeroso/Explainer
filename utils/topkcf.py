from config.explainer import Path, Args

class TopKCounterfactualsTox21:
    Leaderboard = None
    K = 5
    save_dir = None
    @staticmethod
    def init(original, index, save_dir, k=5):
        TopKCounterfactualsTox21.save_dir = save_dir
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

        with open(TopKCounterfactualsTox21.save_dir + '/' + str(TopKCounterfactualsTox21.Leaderboard['index']) + '.json','w') as f:
            json.dump(TopKCounterfactualsTox21.Leaderboard, f, indent=2)


class TopKCounterfactualsESOL:
    Leaderboard = None
    K = 5
    save_dir = None

    @staticmethod
    def init(original, index, save_dir, k=5):

        TopKCounterfactualsESOL.save_dir = save_dir
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
        with open(TopKCounterfactualsESOL.save_dir + '/' + str(TopKCounterfactualsESOL.Leaderboard['index']) + '.json', 'w') as f:
            json.dump(TopKCounterfactualsESOL.Leaderboard, f, indent=2)
