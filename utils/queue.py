class SortedQueue:
    def __init__(self,
                 num_items,
                 sort_predicate=None
    ):
        self.num_items = num_items
        self.sort_predicate = sort_predicate
        self.data_ = []

    def contains(self, smiles):
        return any(d['smiles'] == smiles for d in self.data_)

    def insert(self, data):
        assert 'smiles' in data
        assert 'score' in data

        if self.contains(data['smiles']):
            return

        self.data_.append(data)
        self.data_.sort(key=self.sort_predicate, reverse=True)
        self.data_ = self.data_[:self.num_items]

    def extend(self, queue):
        assert isinstance(queue, SortedQueue)

        for data in queue.data_:
            self.insert(data)
