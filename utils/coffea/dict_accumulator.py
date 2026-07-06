from coffea import processor


class DictAccumulator(processor.AccumulatorABC):
    def __init__(self, value={}):
        self.value = value

    def identity(self):
        return DictAccumulator({})
    
    def add(self, other):
        if len(other.value.keys()) == 0:
            return
        elif len(self.value.keys()) == 0:
            self.value = other.value
        else:
            # Keys are not required to match: e.g. norm_lund's keys are per
            # n-prong (lundWeightNom_8_sumw, lundWeightNom_10_sumw, ...) and
            # different chunks/files can have different prong multiplicities
            # present. Keys present in both are summed; keys present in only
            # one side are carried over as-is.
            for key in other.value.keys():
                if key in self.value:
                    self.value[key] += other.value[key]
                else:
                    self.value[key] = other.value[key]

