import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score

class TDCDDIEvaluator:
    def __init__(self, name):
        self.name = name
        self.eval_metric = 'rocauc'  # Default evaluation metric for DDI

    def _parse_and_check_input(self, input_dict):
        if 'y_pred_pos' not in input_dict:
            raise RuntimeError('Missing key of y_pred_pos')
        if 'y_pred_neg' not in input_dict:
            raise RuntimeError('Missing key of y_pred_neg')

        y_pred_pos, y_pred_neg = input_dict['y_pred_pos'], input_dict['y_pred_neg']

        # Convert y_pred_pos, y_pred_neg into either torch tensor or numpy array
        type_info = None

        if torch is not None and (isinstance(y_pred_pos, torch.Tensor) or isinstance(y_pred_neg, torch.Tensor)):
            if isinstance(y_pred_pos, np.ndarray):
                y_pred_pos = torch.from_numpy(y_pred_pos)
            if isinstance(y_pred_neg, np.ndarray):
                y_pred_neg = torch.from_numpy(y_pred_neg)
            y_pred_pos = y_pred_pos.to(y_pred_neg.device)
            type_info = 'torch'
        else:
            type_info = 'numpy'

        if not y_pred_pos.ndim == 1:
            raise RuntimeError('y_pred_pos must be 1-dim array, {}-dim array given'.format(y_pred_pos.ndim))
        if not y_pred_neg.ndim == 1:
            raise RuntimeError('y_pred_neg must be 1-dim array, {}-dim array given'.format(y_pred_neg.ndim))

        return y_pred_pos, y_pred_neg, type_info

    def eval(self, input_dict):

        y_pred_pos, y_pred_neg, type_info = self._parse_and_check_input(input_dict)
        return self._eval_metrics(y_pred_pos, y_pred_neg, type_info)

    @property
    def expected_input_format(self):
        desc = '==== Expected input format of Evaluator for {}\n'.format(self.name)
        desc += '{\'y_pred_pos\': y_pred_pos, \'y_pred_neg\': y_pred_neg}\n'
        desc += '- y_pred_pos: numpy ndarray or torch tensor of shape (num_edges, ).\n'
        desc += '- y_pred_neg: numpy ndarray or torch tensor of shape (num_edges, ).\n'
        desc += 'y_pred_pos is the predicted scores for positive edges.\n'
        desc += 'y_pred_neg is the predicted scores for negative edges.\n'
        return desc

    @property
    def expected_output_format(self):
        desc = '==== Expected output format of Evaluator for {}\n'.format(self.name)
        desc += '{\'rocauc\': rocauc, \'prauc\': prauc, \'accuracy\': accuracy, \'hits@1\': hits@1, \'hits@3\': hits@3, \'hits@10\': hits@10, \'mrr\': mrr}\n'
        desc += '- rocauc (float): ROC-AUC score\n'
        desc += '- prauc (float): Precision-Recall AUC score\n'
        desc += '- accuracy (float): Accuracy score\n'
        desc += '- hits@1 (float): Hits@1 score\n'
        desc += '- hits@3 (float): Hits@3 score\n'
        desc += '- hits@10 (float): Hits@10 score\n'
        desc += '- mrr (float): Mean Reciprocal Rank score\n'
        return desc

    def _eval_metrics(self, y_pred_pos, y_pred_neg, type_info):
        if len(y_pred_pos) == 0 or len(y_pred_neg) == 0:
            return {
                'rocauc': 0.0,
                'prauc': 0.0,
                'accuracy': 0.0,
                'hits@1': 0.0,
                'hits@3': 0.0,
                'hits@10': 0.0,
                'hits@20': 0.0,
                'hits@30': 0.0,
                'hits@40': 0.0,
                'hits@50': 0.0,
                'mrr': 0.0
            }
        if type_info == 'torch':
            y_pred_pos_numpy = y_pred_pos.cpu().numpy()
            y_pred_neg_numpy = y_pred_neg.cpu().numpy()
        else:
            y_pred_pos_numpy = y_pred_pos
            y_pred_neg_numpy = y_pred_neg

        y_true = np.concatenate([np.ones(len(y_pred_pos_numpy)), np.zeros(len(y_pred_neg_numpy))]).astype(np.int32)
        y_pred = np.concatenate([y_pred_pos_numpy, y_pred_neg_numpy])

        # Compute ROC-AUC
        rocauc = roc_auc_score(y_true, y_pred)

        # Compute Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        prauc = auc(recall, precision)

        # Compute Accuracy
        y_pred_binary = (y_pred >= 0.5).astype(np.int32)
        accuracy = accuracy_score(y_true, y_pred_binary)

        # Compute Hits@K
        hits1 = self._compute_hits(y_pred_pos_numpy, y_pred_neg_numpy, 1)
        hits3 = self._compute_hits(y_pred_pos_numpy, y_pred_neg_numpy, 3)
        hits10 = self._compute_hits(y_pred_pos_numpy, y_pred_neg_numpy, 10)
        hits20 = self._compute_hits(y_pred_pos_numpy, y_pred_neg_numpy, 20)
        hits30 = self._compute_hits(y_pred_pos_numpy, y_pred_neg_numpy, 30)
        hits40 = self._compute_hits(y_pred_pos_numpy, y_pred_neg_numpy, 40)
        hits50 = self._compute_hits(y_pred_pos_numpy, y_pred_neg_numpy, 50)

        # Compute MRR
        mrr = self._compute_mrr(y_pred_pos_numpy, y_pred_neg_numpy)

        return {
            'rocauc': rocauc,
            'prauc': prauc,
            'accuracy': accuracy,
            'hits@1': hits1,
            'hits@3': hits3,
            'hits@10': hits10,
            'hits@20': hits20,
            'hits@30': hits30,
            'hits@40': hits40,
            'hits@50': hits50,
            'mrr': mrr
        }

    def _compute_hits(self, y_pred_pos, y_pred_neg, K):
        if len(y_pred_neg) < K:
            return 1.0
        # More efficient: get K-th largest negative score
        kth_score = np.partition(y_pred_neg, -K)[-K]
        # Optional: use >= if you want ties to count
        hitsK = float(np.sum(y_pred_pos > kth_score)) / len(y_pred_pos)
        return hitsK

    def _compute_mrr(self, y_pred_pos, y_pred_neg):
        all_preds = np.concatenate([y_pred_pos, y_pred_neg])
        all_true = np.concatenate([np.ones_like(y_pred_pos), np.zeros_like(y_pred_neg)])
        ranking = np.argsort(-all_preds)  # Sort in descending order
        mrr_list = []
        for pos_score in y_pred_pos:
            # Find rank of the positive sample (1-based)
            rank = np.where(ranking == np.where(all_true == 1)[0][0])[0][0] + 1
            mrr_list.append(1.0 / rank)
        return np.mean(mrr_list)

def run_tests():
    evaluator = TDCDDIEvaluator(name='TDC_DDI')
    test_cases = {
        "Perfect Predictions": {'y_pred_pos': np.array([0.9, 0.8, 0.7]), 'y_pred_neg': np.array([0.2, 0.3, 0.4])},
        "Worst Predictions": {'y_pred_pos': np.array([0.2, 0.3, 0.4]), 'y_pred_neg': np.array([0.9, 0.8, 0.7])},
        "Random Predictions": {'y_pred_pos': np.array([0.6, 0.5, 0.4]), 'y_pred_neg': np.array([0.5, 0.3, 0.7])},
        "Empty Inputs": {'y_pred_pos': np.array([]), 'y_pred_neg': np.array([0.1, 0.2])},
        "Equal Scores": {'y_pred_pos': np.array([0.5, 0.5]), 'y_pred_neg': np.array([0.5, 0.5])},
        "Varying K": {'y_pred_pos': np.array([0.9, 0.8, 0.7]), 'y_pred_neg': np.array([0.85, 0.75, 0.6, 0.5, 0.4])},
        "MRR Check": {'y_pred_pos': np.array([0.7]), 'y_pred_neg': np.array([0.9, 0.8])},
        "Partial Correctness": {'y_pred_pos': np.array([0.9, 0.4, 0.6]), 'y_pred_neg': np.array([0.5, 0.3, 0.8])}
    }
    for name, input_dict in test_cases.items():
        print(f"\n=== Test Case: {name} ===")
        result = evaluator.eval(input_dict)
        print(result)

if __name__ == "__main__":
    run_tests()

