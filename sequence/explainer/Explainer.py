from GradientExplainer.GradientExplainer_masking import *
import shap
from lime import lime_text


class Explainer:

    def __init__(self, model, method, **kwargs):
        '''
        create explainer
        -model: for gradient-based methods, only support layer {bi-lstm, bi-gru, masking, dense}
                for shap & lime, support most common models, see more details in shap & lime's github repository
                https://github.com/marcotcr/lime  https://github.com/slundberg/shap
        -method: now only support ['GradInput', 'LRP', 'DeepLift', 'SHAP', 'LIME'] for vul detect models
        -kwargs: gradient-based methods: None
                 shap: 'shap_strategy': three shap's calc strategy 'Deep', 'Gradient', 'Kernel'
                       'shap_background': shap's reference datasets, typically choose 50-200 samples, see more details in shap's api
                 lime: 'kernel_width' & 'feature_selection' are the important hyper-para might need to optimize
                       'split_expression': the expression you use in spliting text into tokens, either func or regular express
                       'bow' & 'char_level' should be fixed for most vul detect models
                       see other parameters' details in lime's api
        '''

        self.model = model
        self.method = method

        if method not in ['GradInput', 'LRP', 'DeepLift', 'SHAP', 'LIME']:
            raise Exception('invalid method name')

        if self.method == 'SHAP':
            self.shap_strategy = kwargs['shap_strategy'] if 'shap_strategy' in kwargs else 'Deep'
            if 'shap_background' in kwargs:
                self.shap_background = kwargs['shap_background']
            else:
                raise Exception('SHAP method need to provide a background')
            if self.shap_strategy == 'Deep':
                self.explainer = shap.DeepExplainer(self.model, self.shap_background)
            elif self.shap_strategy == 'Gradient':
                self.explainer = shap.GradientExplainer(self.model, self.shap_background)
            elif self.shap_strategy == 'Kernel':
                self.explainer = shap.KernelExplainer(self.model, self.shap_background)
            else:
                raise Exception('use wrong shap strategy, should be {"Deep", "Gradient", "Kernel"}')
        elif self.method == 'LIME':
            self.lime_split_expression = r'\\W+' if 'lime_split_expression' not in kwargs else kwargs['lime_split_expression']
            self.explainer = lime_text.LimeTextExplainer\
                (kernel_width=25            if 'lime_kernel_width'      not in kwargs else kwargs['lime_kernel_width'],
                 kernel=None                if 'lime_kernel'            not in kwargs else kwargs['lime_kernel'],
                 verbose=False              if 'lime_verbose'           not in kwargs else kwargs['lime_verbose'],
                 class_names=None           if 'lime_class_names'       not in kwargs else kwargs['lime_class_names'],
                 feature_selection='auto'   if 'lime_feature_selection' not in kwargs else kwargs['lime_feature_selection'],
                 split_expression=self.lime_split_expression,
                 bow=False,
                 mask_string=None           if 'lime_mask_string'       not in kwargs else kwargs['lime_mask_string'],
                 random_state=None          if 'lime_random_state'      not in kwargs else kwargs['lime_random_state'],
                 char_level=False)
        else:
            self.explainer = GradientExplainer(self.model)


    def explain(self, inputs, target_class=0, **kwargs):
        '''
        Todo the input&output formats are different, but changing shap&lime's code just for formats is too expensive
        explain sample
        -inputs: multi data input with batch
                 for lime, the calc function need to take raw text as input, now directly use the lime's api
        -target_class: for gradient-based methods: explain single class's prediction, index 0,1,..., default 0
                       for shap: default 0 explain all class, or explain top-k class, k=target_class, return (shap_values, indexes)
                       for lime: target class is iterable indicate the inputs, default (1,)
        -kwargs: gradient*input: None
                 LRP: 'eps'
                 DeepLift: 'eps', 'inputs_ref'
                 LIME: 'num_features': return top k tokens, k=num_features,
                       'num_samples': sample time, typically 1000-2000 is enough for vul detect model,
                       'distance_metric'
                 SHAP: None

        return: [[(sample1_value,class1), (sample1_value,class2), ...], [(sample2_value,class1), ...]]
                sample_value format: [[token1,position1,weight1], ...]
        '''
        results = []
        if self.method == 'GradInput':
            for input in inputs:
                results.append(self.explainer.gradient_input(input, target_class))
        elif self.method == 'LRP':
            eps = kwargs['eps'] if 'eps' in kwargs else 0.001
            for input in inputs:
                results.append(self.explainer.lrp(input, target_class, eps))
        elif self.method == 'DeepLift':
            eps = kwargs['eps'] if 'eps' in kwargs else 0.001
            inputs_ref = kwargs['inputs_ref'] if 'inputs_ref' in kwargs else 0.0
            for input in inputs:
                results.append(self.explainer.deeplift(input, inputs_ref, target_class, eps))
        elif self.method == 'SHAP':
            if target_class == 0:
                target_class = None
            results = self.explainer.shap_values(inputs, ranked_outputs=target_class)
        else:
            for input in inputs:
                result = self.explainer.explain_instance(input, self.model,
                                                target_class=(1,)           if target_class == 0        else target_class,
                                                num_features=10             if 'lime_num_features'      not in kwargs else kwargs['lime_num_features'],
                                                num_samples=5000            if 'lime_num_samples'       not in kwargs else kwargs['lime_num_samples'],
                                                distance_metric='cosine'    if 'lime_distance_metric'   not in kwargs else kwargs['lime_distance_metric'])
                results.append(self.trans_position(result, input))

        return results


    def trans_position(self, result, text):     # get tokens' position
        result = result.as_list(positions=True)
        result_processed = []
        for tuple in result:
            word_position, weight = tuple[0], tuple[1]
            word_position = word_position.rsplit('_', 1)
            word = word_position[0]     # token
            position = int(word_position[1])    # original position
            if callable(self.lime_split_expression):
                token_list = self.lime_split_expression(text)
            else:
                token_list = text.split(self.lime_split_expression)

            count_position = -1
            transformed_position = 0
            for token in token_list:
                if position == count_position+1 and text[count_position+1:].startwith(word):
                    result_processed.append([word, transformed_position, weight])
                    break
                else:
                    count_position += len(token)
                    while text[count_position+1] == ' ':
                        count_position += 1
                    transformed_position += 1

        return result_processed