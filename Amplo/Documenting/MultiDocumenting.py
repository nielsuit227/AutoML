import os
import copy
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from .BinaryDocumenting import BinaryDocumenting


class MultiDocumenting(BinaryDocumenting):

    def __init__(self, pipeline):
        super().__init__(pipeline)

    def analyse(self):
        # Initiating
        f1_score = np.zeros((self.p.cvSplits, self.p.n_classes))
        log_loss = np.zeros(self.p.cvSplits)
        avg_acc = np.zeros(self.p.cvSplits)
        cm = np.zeros((self.p.cvSplits, self.p.n_classes, self.p.n_classes))

        # Modelling
        model = None
        self.cv = StratifiedKFold(n_splits=self.p.cvSplits, shuffle=self.p.shuffle)
        for i, (t, v) in enumerate(self.cv.split(self.x, self.y)):
            xt, xv, yt, yv = self.x[t], self.x[v], self.y[t].reshape((-1)), self.y[v].reshape((-1))
            model = copy.copy(self.model)
            model.fit(xt, yt)
            predictions = model.predict(xv).reshape((-1))
            cm[i] = metrics.confusion_matrix(predictions, yv)

            # Metrics
            f1_score[i] = metrics.f1_score(yv, predictions, average=None)
            avg_acc[i] = metrics.accuracy_score(yv, predictions)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(xv)
                log_loss[i] = metrics.log_loss(yv, probabilities)

        # Result statistics
        totals = np.sum(cm, axis=(1, 2), keepdims=True)
        means = np.mean(cm / totals * 100, axis=0)
        stds = np.std(cm / totals * 100, axis=0)

        # Store
        self.metrics = {
            'F1 Score': [np.mean(f1_score), np.std(f1_score)],
            'Accuracy': [np.mean(avg_acc), np.std(avg_acc)],
        }
        self.outputMetrics = copy.deepcopy(self.metrics)
        self.outputMetrics['Confusion Matrix Means'] = means
        self.outputMetrics['Confusion Matrix Stds'] = stds
        self.confusion_matrix = {
            'means': means,
            'stds': stds,
        }

        # Print
        print('F1 scores:')
        print(''.join([' Class {} |'.format(i) for i in range(self.p.n_classes)]))
        print(''.join([' {:.2f} % '.ljust(11).format(f1) + '|' for f1 in np.mean(f1_score, axis=0)]))
        print('Average Accuracy: {:.2f} \u00B1 {:.2f} %'.format(np.mean(avg_acc), np.std(avg_acc)))
        if hasattr(model, 'predict_proba'):
            print('Log Loss:         {:.2f} \u00B1 {:.2f}'.format(np.mean(log_loss), np.std(log_loss)))
            self.metrics['Log Loss'] = [np.mean(log_loss), np.std(log_loss)]

        if not os.path.exists(self.p.mainDir + 'EDA/Features/v{}/RF.png'.format(self.p.version)):
            if not os.path.exists(self.p.mainDir + 'EDA/Features/v{}'.format(self.p.version)):
                os.makedirs(self.p.mainDir + 'EDA/Features/v{}/'.format(self.p.version))
            fig, ax = plt.subplots(figsize=[4, 6], constrained_layout=True)
            plt.subplots_adjust(left=0.5, top=1, bottom=0)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['top'].set_visible(False)
            keys, fi = self.p.featureProcesser.featureImportance['rf']
            plt.barh(keys[:15], width=fi[:15], color='#2369ec')
            fig.savefig(self.p.mainDir + 'EDA/Features/v{}/RF.png'.format(self.p.version), format='png', dpi=200)

    def model_performance(self):
        self.ln(self.lh)
        self.add_h2('Model Performance')
        self.add_text('Model performance is analysed by various metrics. This model has been selected based on the {} '
                      'score.'.format(self.p.objective))

        # Metrics
        self.set_font('Helvetica', 'B', 12)
        self.ln(self.lh)
        self.cell(w=50, h=self.lh, txt='Metric', border='B', align='C')
        self.cell(w=50, h=self.lh, txt='Score', border='LB', align='C')
        self.set_font('Helvetica', '', 12)
        for k, v in self.metrics.items():
            self.ln(self.lh)
            self.cell(w=50, h=self.lh, txt=k, border='R', align='L')
            self.cell(w=50, h=self.lh, txt='{:.2f} \u00B1 {:.2f} %'.format(v[0], v[1]), border='L', align='C')
        self.ln(self.lh * 3)

        # Confusion Matrix
        nc = self.p.n_classes
        cell_width = int((self.WIDTH - self.pm * 2) / (nc + 2))
        self.add_h3('Confusion Matrix')

        # First row
        self.set_font('Helvetica', 'B', 12)
        self.cell(w=cell_width * 2, h=self.lh, txt='', align='L', border='R')
        self.cell(w=cell_width * nc, h=self.lh, txt='True Class', align='C')
        self.ln(self.lh)

        # Second Row
        self.cell(w=cell_width * 2, h=self.lh, txt='', align='L', border='B')
        for i in range(nc):
            self.cell(w=cell_width, h=self.lh, txt='Class {}'.format(i), align='C', border='BL')
        self.ln(self.lh)

        # Values
        for i in range(nc):
            # Set bold
            self.set_font('Helvetica', 'B', 12)
            # Prediction (only for first time)
            if i == 0:
                self.cell(w=cell_width, h=self.lh * nc, txt='Prediction', align='L')
            else:
                self.cell(w=cell_width, h=self.lh, txt='')
            # Class
            self.cell(w=cell_width, h=self.lh, txt='Class {}'.format(i), align='L', border='R')
            # Set normal
            self.set_font('Helvetica', '', 12)
            for j in range(nc):
                self.cell(w=cell_width, h=self.lh, txt='{:.1f}\u00B1{:.1f} %'.format(
                    self.confusion_matrix['means'][i][j], self.confusion_matrix['stds'][i][j]), align='C')
            self.ln(self.lh)

    def validation(self):
        self.ln(self.lh)
        self.check_new_page()
        self.add_h3('Validation Strategy')
        self.add_text("All experiments are cross-validated. This means that every time a model's "
                      "performance is evaluated, it's trained on one part of the data, and test on another. Therefore, "
                      "the model is always test against data it has not yet been trained for. This gives the best "
                      "approximation for real world (out of sample) performance. The current validation strategy used "
                      "is {}, with {} splits and {} shuffling the data."
                      .format(type(self.cv).__name__, self.p.cvSplits, 'with' if self.p.shuffle else 'without'))
