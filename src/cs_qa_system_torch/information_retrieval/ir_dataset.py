import logging


from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class InferenceDataset(Dataset):
    def __init__(self, predict_list, query_transform, context_transform):
        self.predict_list = predict_list
        self.query_transform = query_transform
        self.context_transform = context_transform

    def __len__(self):
        return len(self.predict_list)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self.__get_single_item__(index) for index in indices]
        return self.__get_single_item__(indices)

    def __get_single_item__(self, index):
        qid, pid, query, context, score = self.predict_list[index]
        transformed_query = self.query_transform(query)  # [token_ids],[seg_ids],[masks]
        transformed_context = self.context_transform(context)  # [token_ids],[seg_ids],[masks]
        return (*transformed_query, *transformed_context)


class CombinedInferenceDataset(Dataset):
    def __init__(self, predict_list, transform):
        self.predict_list = predict_list
        self.transform = transform

    def __len__(self):
        return len(self.predict_list)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self.__get_single_item__(index) for index in indices]
        return self.__get_single_item__(indices)

    def __get_single_item__(self, index):
        qid, pid, query, context, score = self.predict_list[index]
        transformed_query_context = self.transform(query, context)  # [token_ids],[seg_ids],[masks]
        return transformed_query_context