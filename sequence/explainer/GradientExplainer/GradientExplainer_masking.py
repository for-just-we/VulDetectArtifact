import numpy as np
from layers_masking import Masking, Liner, Bi_GRU, Bi_LSTM


class GradientExplainer:

    def __init__(self, model, bias_factor=0.0):
        """
        -model: keras model, now only support layer: {Bi_LSTM, Bi_GRU, Liner, Masking}
        -prediction_class: choose one class to calc scores, start from 0
        -bias_factor: check sanity but need extra calculation
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
                                              recurrent_activation=layer_conf['layer']['config']['recurrent_activation']))      # Todo: automatic activation
            elif layer_conf['name'].startswith('dense'):
                self.layer_name_list.append('Liner')
                self.layer_kwargs_list.append(dict(liner_activation=layer_conf['activation']))
            elif layer_conf['name'].startswith('masking'):
                self.layer_name_list.append('Masking')
                self.layer_kwargs_list.append(dict(mask_value=layer_conf['mask_value']))
            else:
                raise Exception(f'layer {layer_conf["name"]} currently not support')


    def forward(self, inputs):
        """
        perform forward pass for sanity check

        -inputs: model inputs without batch
        """
        mask = np.full((inputs.shape[0],), False)       # shape (T,)
        layer_results = inputs
        for i,layer in enumerate(self.model.layers):
            layer_method = globals()[self.layer_name_list[i]]
            layer_results, inter_results = layer_method.forward(layer_results, layer.get_weights(), mask, **self.layer_kwargs_list[i])
            if self.layer_name_list[i] == 'Masking':    # suppose only masking layer can change self.mask
                mask = inter_results['mask']

        return layer_results


    def lrp(self, inputs, prediction_class=0, eps=0.001, bias_factor=0.0):
        """
        perform lrp

        -inputs: model inputs without batch
        -prediction_class: choose one class to calc scores, start from 0
        -eps: lrp rule
        -bias_factor: for sanity check
        """
        mask_list = [np.full((inputs.shape[0],), False)]        # stored all mask for every layer
        self.layer_kwargs_list[-1]['LRP_class'] = prediction_class
        layer_results_list = [inputs]
        for i,layer in enumerate(self.model.layers):
            layer_method = globals()[self.layer_name_list[i]]
            layer_results, inter_results = layer_method.forward(layer_results_list[i], layer.get_weights(), mask_list[i], **self.layer_kwargs_list[i])
            layer_results_list.append(layer_results)
            if self.layer_name_list[i] == 'Masking':
                mask_list.append(inter_results['mask'])       # suppose only masking layer can change self.mask
            else:
                mask_list.append(mask_list[-1])

        R = inter_results['scores']                           # Todo: now only liner layer can return scores
        for i in reversed(range(len(self.model.layers))):
            layer_method = globals()[self.layer_name_list[i]]
            R, relevance_sum = layer_method.lrp(layer_results_list[i], self.model.layers[i].get_weights(), R, mask_list[i], eps, bias_factor, **self.layer_kwargs_list[i])

        return R, relevance_sum



    def gradient_input(self, inputs, prediction_class=0, bias_factor=0.0):
        """
        perform gradient*input
        as mentioned in https://link.springer.com/chapter/10.1007/978-3-030-28954-6_10, gradient*input = LRP-0

        -inputs: model inputs without batch
        -prediction_class: choose one class to calc scores, start from 0
        -bias_factor: for sanity check
        """
        mask_list = [np.full((inputs.shape[0],), False)]  # stored all mask for every layer
        self.layer_kwargs_list[-1]['LRP_class'] = prediction_class
        layer_results_list = [inputs]
        for i, layer in enumerate(self.model.layers):
            layer_method = globals()[self.layer_name_list[i]]
            layer_results, inter_results = layer_method.forward(layer_results_list[i], layer.get_weights(), mask_list[i], **self.layer_kwargs_list[i])
            layer_results_list.append(layer_results)
            if self.layer_name_list[i] == 'Masking':
                mask_list.append(inter_results['mask'])  # suppose only masking layer can change self.mask
            else:
                mask_list.append(mask_list[-1])

        R = inter_results['scores']
        for i in reversed(range(len(self.model.layers))):
            layer_method = globals()[self.layer_name_list[i]]
            R, relevance_sum = layer_method.lrp(layer_results_list[i], self.model.layers[i].get_weights(), R, mask_list[i], eps=0.0, bias_factor=bias_factor, **self.layer_kwargs_list[i])

        return R, relevance_sum


    def deeplift(self, inputs, inputs_ref=0.0, prediction_class=0, eps=0.001, bias_factor=0.0):             # Todo: need farther verify
        """
        perform lrp

        -inputs: model inputs without batch
        -inputs_ref: default all zeros, if add another reference, the shape should equal to inputs
        -prediction_class: choose one class to calc scores, start from 0
        -eps: lrp rule
        -bias_factor: for sanity check

        4.29 change: when masking is 0.0 and ref also set to 0.0, the whole ref would be masked, so the input_ref should have same mask_list when input
        """

        mask_list = [np.full((inputs.shape[0],), False)]  # stored all mask for every layer
        self.layer_kwargs_list[-1]['LRP_class'] = prediction_class
        layer_results_list = [inputs]
        if inputs_ref == 0.0:
            inputs_ref = np.zeros_like(inputs)
            # if self.layer_kwargs_list[0]['mask_value'] == 0:    # if mask value == ref value, then change ref to avoid all zeros
            #     inputs_ref = np.ones_like(inputs)
        layer_results_ref_list = [inputs_ref]
        for i,layer in enumerate(self.model.layers):
            layer_method = globals()[self.layer_name_list[i]]
            layer_results, inter_results = layer_method.forward(layer_results_list[i], layer.get_weights(), mask_list[i], **self.layer_kwargs_list[i])
            layer_results_ref, inter_results_ref = layer_method.forward(layer_results_ref_list[i], layer.get_weights(), mask_list[i], **self.layer_kwargs_list[i])
            layer_results_list.append(layer_results)
            layer_results_ref_list.append(layer_results_ref)
            if self.layer_name_list[i] == 'Masking':
                mask_list.append(inter_results['mask'])  # suppose only masking layer can change self.mask
                inter_results_ref['mask'] = inter_results['mask']
            else:
                mask_list.append(mask_list[-1])

        R = inter_results['scores'] - inter_results_ref['scores']
        for i in reversed(range(len(self.model.layers))):
            layer_method = globals()[self.layer_name_list[i]]
            R, relevance_sum = layer_method.deeplift(layer_results_list[i], layer_results_ref_list[i], self.model.layers[i].get_weights(),
                                                     R, mask_list[i], eps, bias_factor, **self.layer_kwargs_list[i])

        return R, relevance_sum




###################################
# Todo list:
##################################
"""
1 add support to more activation functions
2 may be extend the other lrp rules, e.g. lrp-y
3 clear redundant calculations
4 add sanity check, every lrp() should calc and pass relevance of other neurons
5 farther verify deeplift, may be use the original codes    (should be ok)
6 farther verify Masking() in Bi-RGU Bi-LSTM                (should be ok)
7 optimize the structure of layers.py, now the masking, return_sequences and lrp propagation are calculated together
8 support models from pytorch, tensorflow | support more layers | support more layers' customized parameters
9 farther verify whether gradient*input is equal to lrp-o 
"""