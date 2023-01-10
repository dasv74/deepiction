import os
from deepiction.manager import load_model, predict_model, save_predicted_images, is_tensorflow, is_pytorch
from deepiction.manager import compute_metrics, figure_images, figure_histo, parameters
import datetime
import pickle
import tensorflow

class Test:

    def __init__(self, model):
        if type(model) == str:
            self.model = load_model(model)   
        else:
            self.model = model
            
    def run(self, dataset, metricname, path):
        report_framework = 'unknown'
        if is_tensorflow(self.model): report_framework = 'TensorFlow ' + tensorflow. __version__
        if is_pytorch(self.model): report_framework = 'Pytorch '
        occ = self.occurence(path, 'report_test')
        report = {
            'Framework': report_framework,
            'Model': f'{type(self.model)} weights: {parameters(self.model)}',
            'Occurence': occ,
            'Dataset': dataset.name,
            'Starting time':  datetime.datetime.now(),
        }
        self.predictions = predict_model(self.model, dataset.sources)
        report['Ending time'] = datetime.datetime.now()
        save_predicted_images(dataset, self.predictions, path)
        metrics = compute_metrics(dataset, self.predictions, metricname, path=path)
        figure_histo(path, metrics, metricname, occ)
        figure_images(dataset, self.predictions, 5, path, metricname, metrics, occ)
        with open(os.path.join(path, f'report_test_{occ}.pickle'), 'wb') as f:
            pickle.dump(report, f)
        f.close()
        return report
    
    def occurence(self, output_path, pattern):
        c = 0
        for f in os.listdir(output_path):
            if os.path.isfile(os.path.join(output_path, f)):
                if f.startswith(pattern):
                    c += 1
        return str(c)  
        
