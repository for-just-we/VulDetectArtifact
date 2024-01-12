import numpy as np

class GradientExplainer:

    def __init__(self, model, bias_factor=0.0):
        """
        -model: keras model, now only support layer: {bi-lstm, bi-gru, liner}
        -bias_factor: check sanity but need extra calculation,
            and attention if the relevance scores sum up not strictly equals to prediction score, probably because the bias
        """
        self.model              = model
        self.bias_factor        = bias_factor
        self.layer_name_list    = []
        self.layer_kwargs_list  = []

        for layer in model.layers:
            layer_conf = layer.get_config()
            if layer_conf['name'].startswith('bidirectional') and layer_conf['layer']['class_name'] in ['GRU','LSTM']:
                self.layer_name_list.append('Bi_'+layer_conf['layer']['class_name'])
                self.layer_kwargs_list.append(dict(return_sequences=layer_conf['layer']['config']['return_sequences'],
                                              recurrent_activation=layer_conf['layer']['config']['recurrent_activation']))
            elif layer_conf['name'].startswith('dense'):
                self.layer_name_list.append('Liner')
                self.layer_kwargs_list.append(dict(liner_activation=layer_conf['activation']))
            else:
                raise Exception(f'layer {layer_conf["name"]} currently not support')


    def forward(self, inputs):
        """
        perform forward pass for sanity check

        -inputs: model inputs without batch
        """
        layer_results = inputs
        for i,layer in enumerate(self.model.layers):
            layer_method = globals()[self.layer_name_list[i]]
            layer_results, inter_results = layer_method.forward(layer_results, layer.get_weights(), **self.layer_kwargs_list[i])

        return layer_results


    def lrp(self, inputs, prediction_class=0, eps=0.001, bias_factor=0.0):
        """
        perform lrp

        -inputs: model inputs without batch
        -prediction_class: choose one class to calc scores, start from 0
        -eps: lrp rule
        -bias_factor: for sanity check
        """
        self.layer_kwargs_list[-1]['LRP_class'] = prediction_class
        layer_results_list = [inputs]
        for i,layer in enumerate(self.model.layers):
            layer_method = globals()[self.layer_name_list[i]]
            layer_results, inter_results = layer_method.forward(layer_results_list[i], layer.get_weights(), **self.layer_kwargs_list[i])
            layer_results_list.append(layer_results)

        R = inter_results['scores']
        for i in reversed(range(len(self.model.layers))):
            layer_method = globals()[self.layer_name_list[i]]
            R, relevance_sum = layer_method.lrp(layer_results_list[i], self.model.layers[i].get_weights(), R, eps, bias_factor, **self.layer_kwargs_list[i])

        return R, relevance_sum


    def gradient_input(self, inputs, prediction_class=0, bias_factor=0.0):
        """
        perform gradient*input
        as mentioned in https://link.springer.com/chapter/10.1007/978-3-030-28954-6_10, gradient*input = LRP-0

        -inputs: model inputs without batch
        -prediction_class: choose one class to calc scores, start from 0
        -bias_factor: for sanity check
        """
        self.layer_kwargs_list[-1]['LRP_class'] = prediction_class
        layer_results_list = [inputs]
        for i, layer in enumerate(self.model.layers):
            layer_method = globals()[self.layer_name_list[i]]
            layer_results, inter_results = layer_method.forward(layer_results_list[i], layer.get_weights(), **self.layer_kwargs_list[i])
            layer_results_list.append(layer_results)

        R = inter_results['scores']
        for i in reversed(range(len(self.model.layers))):
            layer_method = globals()[self.layer_name_list[i]]
            R, relevance_sum = layer_method.lrp(layer_results_list[i], self.model.layers[i].get_weights(), R, eps=0.0, bias_factor=bias_factor, **self.layer_kwargs_list[i])

        return R, relevance_sum


    def deeplift(self, inputs, inputs_ref=0.0, prediction_class=0, eps=0.001, bias_factor=0.0):
        """
        perform lrp

        -inputs: model inputs without batch
        -inputs_ref: default all zeros
        -prediction_class: choose one class to calc scores, start from 0
        -eps: lrp rule
        -bias_factor: for sanity check
        """

        self.layer_kwargs_list[-1]['LRP_class'] = prediction_class
        layer_results_list = [inputs]
        if inputs_ref == 0.0:
            inputs_ref = np.zeros_like(inputs)
        layer_results_ref_list = [inputs_ref]
        for i,layer in enumerate(self.model.layers):
            layer_method = globals()[self.layer_name_list[i]]
            layer_results, inter_results = layer_method.forward(layer_results_list[i], layer.get_weights(), **self.layer_kwargs_list[i])
            layer_results_ref, inter_results_ref = layer_method.forward(layer_results_ref_list[i], layer.get_weights(), **self.layer_kwargs_list[i])
            layer_results_list.append(layer_results)
            layer_results_ref_list.append(layer_results_ref)

        R = inter_results['scores'] - inter_results_ref['scores']
        for i in reversed(range(len(self.model.layers))):
            layer_method = globals()[self.layer_name_list[i]]
            R, relevance_sum = layer_method.deeplift(layer_results_list[i], layer_results_ref_list[i],
                                                     self.model.layers[i].get_weights(), R, eps, bias_factor, **self.layer_kwargs_list[i])

        return R, relevance_sum




###################################
# Todo list:
##################################
"""
1 verify GradientExplainer
    add automatic activation function
    may be extend the other lrp rules, at least for lrp-y
    clear redundant calculations
2 add gradient*input & deeplift
    farther verify deeplift 
3 add masking
"""