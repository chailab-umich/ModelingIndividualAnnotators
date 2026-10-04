import operator

class AnnotatorMapper:
    def __init__(self, annotators):
        print('Creating annotator mapper')
        self.map_to_annotator = {idx: e for idx, e in enumerate(sorted(annotators))} # Sorted should ensure that dictionary is remade correctly when model is loaded - since annotators is a set sorted is sufficient as there are no duplicate values
        self.map_to_idx = {e: idx for idx, e in self.map_to_annotator.items()}
        self.active_getter = 'map_to_annotator'

    def set_get_idx(self):
        self.active_getter = 'map_to_idx'

    def set_get_annotator(self):
        self.active_getter = 'map_to_annotator'

    def get_annotators(self):
        return set(self.map_to_annotator.values())

    def get_num_annotators(self):
        return len(self.map_to_idx)

    def add_special_annotators(self, aggregate_annotator, self_annotator):
        curr_idx = max(self.map_to_annotator.keys()) + 1
        if aggregate_annotator and 'aggregate-annotator' not in self.map_to_idx:
            self.map_to_idx['aggregate-annotator'] = curr_idx
            self.map_to_annotator[curr_idx] = 'aggregate-annotator'
            curr_idx += 1
        if self_annotator and 'self-report' not in self.map_to_idx:
            self.map_to_idx['self-report'] = curr_idx
            self.map_to_annotator[curr_idx] = 'self-report'
            curr_idx += 1

    def __getitem__(self, l):
        if type(l) == str or type(l) == int:
            return operator.itemgetter(l)(getattr(self, self.active_getter))
        if len(l) == 1:
            return [operator.itemgetter(l[0])(getattr(self, self.active_getter))]
        return list(operator.itemgetter(*l)(getattr(self, self.active_getter)))
