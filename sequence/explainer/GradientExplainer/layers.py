from abc import ABCMeta, abstractmethod
from LRP_linear_layer import *

class layers:
    __meta_class__ = ABCMeta

    """
    now support layer: {'bi-lstm', 'bi-gru', 'liner'}
    """

    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def forward(inputs, weights, **kwargs):
        """
        run forward propagation for one layer

        -inputs: inputs for this layer without batch
        -weights: layer's weights
        -kwargs: layer-specific

        return:
            output: output of this layer without batch
            intermediate_results: dict: stored some layer specific intermediate reuslts
        """
        return

    @staticmethod
    @abstractmethod
    def lrp(inputs, weights, R, eps, bias_factor, **kwargs):
        """
        Update the hidden layer relevance by performing LRP for the target class LRP_class

        -inputs: layer's inputs
        -weights: layer's weights
        -R: relevance from the layer before
        -eps: LRP rule para
        -bias_factor: bias_factor=1.0 for sanity check
        -kwargs: layer specific

        return :
            every neuron's relevance, counterpart with layer's inputs
            relevance_sum: relevance of other neurons, for sanity check
        """
        return

    # @staticmethod
    # @abstractmethod
    # def gradient_input(inputs, weights, R, eps, bias_factor, **kwargs):
    #     """
    #     parameters same as lrp()
    #     """
    #     return

    @staticmethod
    @abstractmethod
    def deeplift(inputs, inputs_ref, weights, R, eps, bias_factor, **kwargs):
        """
        deeplift can be calculated through the lrp backward procedure,
        just calc a reference value for every neuron
        1 do Rin - Rin_ref when call lrp_liner()
        2 do Rout - Rout_ref when call lrp_liner()
        3 do score - score_ref in the top layer
        reference paper: Interpreting Deep Learning Models in Natural Language Processing: A Review

        -inputs_ref: the reference inputs
        -the other parameters are same as lrp
        """
        return



class Liner(layers):

    def __init__(self):
        super(Liner, self).__init__()


    @staticmethod
    def forward(inputs, weights, **kwargs):
        """
        kwargs:
            liner_activation: activation for liner layer

        return:
            intermediate_results: dict: "scores" stored prediction scores without activation
        """
        liner_weights, bias = weights
        scores = np.dot(liner_weights.T, inputs) + bias
        if 'liner_activation' in kwargs.keys():
            if kwargs['liner_activation'] == 'sigmoid':
                y = 1 / (1 + np.exp(-scores))          # Todo***********************
        else:
            y = np.exp(scores) / np.sum(np.exp(scores))     # softmax

        intermediate_results = dict(scores=scores)
        return y, intermediate_results


    @staticmethod
    def lrp(inputs, weights, R, eps, bias_factor, **kwargs):
        """
        kwargs
            LRP_class: if perform lrp for last layer, you must appoint one class's prediction scores to backward, default backward all neurons' scores
        """

        d = inputs.shape[0]//2
        intermediate_results = Liner.forward(inputs, weights, **kwargs)[1]
        outputs = intermediate_results['scores']

        # process bias in dense layer
        if len(weights) == 2:
            liner_weights, bias = weights
        else:
            liner_weights = weights[0]
            bias = np.zeros((liner_weights.shape[0]))

        # if this is the top layer, mask other neuron's prediction
        if 'LRP_class' in kwargs:
            Rout_mask = np.zeros((liner_weights.shape[1]))
            Rout_mask[kwargs['LRP_class']] = 1.0
        else:
            Rout_mask = np.ones((liner_weights.shape[1]))

        Rh = lrp_linear(inputs, liner_weights, bias, outputs, R*Rout_mask, 4*d, eps)

        return Rh, 0.0


    @staticmethod
    def deeplift(inputs, inputs_ref, weights, R, eps, bias_factor, **kwargs):
        """
       kwargs
           LRP_class: if perform lrp for last layer, you must appoint one class's prediction scores to backward, default backward all neurons' scores
       """

        d = inputs.shape[0]//2
        intermediate_results = Liner.forward(inputs, weights, **kwargs)[1]
        outputs = intermediate_results['scores']

        # forward to find R_rev
        intermediate_results_ref = Liner.forward(inputs_ref, weights, **kwargs)[1]
        outputs_ref = intermediate_results_ref['scores']

        # process bias in dense layer
        if len(weights) == 2:
            liner_weights, bias = weights
        else:
            liner_weights = weights[0]
            bias = np.zeros((liner_weights.shape[0]))
        liner_weights = liner_weights.T.reshape(512, 1)

        # if this is the top layer, mask other neuron's prediction
        if 'LRP_class' in kwargs:
            Rout_mask = np.zeros((liner_weights.shape[0]))
            Rout_mask[kwargs['LRP_class']] = 1.0
        else:
            Rout_mask = np.ones((liner_weights.shape[0]))


        inputs  -= inputs_ref
        outputs -= outputs_ref
        R       = lrp_linear(inputs, liner_weights, bias, outputs, R*Rout_mask, 4*d, eps)

        return R, 0.0




class Bi_GRU(layers):

    def __init__(self):
        super(Bi_GRU, self).__init__()


    @staticmethod
    def forward(inputs, weights, **kwargs):
        """
        -kwargs:
            recurrent_activation : activation function for GRU,LSTM, default hard_sigmoid
            return_sequences: same as keras args

        return:
            intermediate_results: dict: "hidden_states" stored all hidden states
        """

         # activation
        def hard_sigmoid(vec):
            zeros = np.zeros_like(vec)
            ones = zeros + 1
            return np.maximum(zeros, np.minimum(ones, 0.2 * vec + 0.5))

        if 'recurrent_activation' in kwargs and kwargs['recurrent_activation'] != 'hard_sigmoid':
            recurrent_activation = kwargs['recurrent_activation']                                   # Todo**************************
        else:
            recurrent_activation = hard_sigmoid

        Wxh_Left, Whh_Left, bxh_Left, Wxh_Right, Whh_Right, bxh_Right = weights                     # split weights
        Wxh_Left, Whh_Left, bxh_Left    = Wxh_Left.T, Whh_Left.T, bxh_Left.T
        Wxh_Right, Whh_Right, bxh_Right = Wxh_Right.T, Whh_Right.T, bxh_Right.T
        T                   = inputs.shape[0]                                                       # time step
        d                   = int(Wxh_Left.shape[0]/3)
        idx                 = np.hstack((np.arange(0,d), np.arange(d,2*d))).astype(int)             # indices of the gates z, r
        idx_z, idx_r, idx_h = np.arange(0,d), np.arange(d,2*d), np.arange(2*d,3*d)                  # indices of z, r, h

        # initialize
        gates_xh_Left  = np.zeros((T, 3*d))
        gates_pre_Left = np.zeros((T, 3*d))         # needed by LRP calc, so need to pass to lrp()
        gates_Left     = np.zeros((T, 3*d))         # also needed by LRP, stored gate sign: z, r, h'

        gates_xh_Right = np.zeros((T, 3*d))
        gates_pre_Right= np.zeros((T, 3*d))
        gates_Right    = np.zeros((T, 3*d))

        h_Left         = np.zeros((T+1, d))       # hidden state in every layer, h[-1] means h0
        h_Right        = np.zeros((T+1, d))


        for t in range(T):

            gates_xh_Left[t]   = np.dot(Wxh_Left, inputs[t]) + bxh_Left                                             # Wz*xt Wr*xt Wh*xt
            z_Left = gates_xh_Left[t,idx_z] + np.dot(Whh_Left[idx_z], h_Left[t-1])                                  # Uz*ht-1
            r_Left = gates_xh_Left[t,idx_r] + np.dot(Whh_Left[idx_r], h_Left[t-1])                                  # Ur*ht-1
            hh_Left = gates_xh_Left[t,idx_h] + np.dot(Whh_Left[idx_h], recurrent_activation(r_Left) * h_Left[t-1])  # W*rt*ht-1

            gates_pre_Left[t] = np.concatenate([z_Left, r_Left, hh_Left], axis = -1)
            gates_Left[t, idx] = recurrent_activation(gates_pre_Left[t, idx])
            gates_Left[t, idx_h] = np.tanh(gates_pre_Left[t, idx_h])                                                # get gate sign: z, r, h'
            # h_Left[t]         = (1-gates_Left[t,idx_z])*h_Left[t-1]+gates_Left[t,idx_z]*gates_Left[t,idx_h]       # why propagation not: ht = (1-z)*ht-1 + z*h' ??
            h_Left[t]		  = gates_Left[t,idx_z]*h_Left[t-1]+(1-gates_Left[t,idx_z])*gates_Left[t,idx_h]


            inputs_rev = inputs[::-1, :].copy() # reverse input
            gates_xh_Right[t]  = np.dot(Wxh_Right, inputs_rev[t]) + bxh_Right
            z_Right = gates_xh_Right[t,idx_z] + np.dot(Whh_Right[idx_z], h_Right[t-1])
            r_Right = gates_xh_Right[t,idx_r] + np.dot(Whh_Right[idx_r], h_Right[t-1])
            hh_Right = gates_xh_Right[t,idx_h] + np.dot(Whh_Right[idx_h], recurrent_activation(r_Right) * h_Right[t-1])

            gates_pre_Right[t] = np.concatenate([z_Right, r_Right, hh_Right], axis = -1)
            gates_Right[t, idx] = recurrent_activation(gates_pre_Right[t, idx])
            gates_Right[t, idx_h] = np.tanh(gates_pre_Right[t, idx_h])                                              # get gate sign: z, r, h'
            # h_Right[t]        = (1-gates_Right[t,idx_z])*h_Right[t-1]+gates_Right[t,idx_z]*gates_Right[t,idx_h]
            h_Right[t]		  = gates_Right[t,idx_z]*h_Right[t-1]+(1-gates_Right[t,idx_z])*gates_Right[t,idx_h]


        intermediate_results = dict(hidden_states = np.concatenate((h_Left.copy(),h_Right.copy()), axis=1),          # pass inter results
                                    gates_pre_Left=gates_pre_Left.copy(), gates_pre_Right=gates_pre_Right.copy(),
                                    gates_Left=gates_Left.copy(), gates_Right=gates_Right.copy())

        if 'return_sequences' in kwargs.keys() and kwargs['return_sequences'] == True:
            return np.concatenate((h_Left[:T].copy(),h_Right[:T][::-1,:].copy()), axis=1), intermediate_results
        else:
            return np.concatenate((h_Left[T-1],h_Right[T-1]), axis=0), intermediate_results


    @staticmethod
    def lrp(inputs, weights, R, eps=0.001, bias_factor=0.0, **kwargs):
        """
        kwargs
            recurrent_activation : activation function for GRU,LSTM, default hard_sigmoid
            return_sequences: same as Keras' return_sequences if True, the Rh should be (T,2*d), else (2*d,)
        """

        intermediate_results    = Bi_GRU.forward(inputs, weights, **kwargs)[1]     # forward pass, attention, need to optimize
        h                       = intermediate_results['hidden_states']
        h_Left, h_Right         = h[:,:h.shape[1]//2], h[:,h.shape[1]//2:]                  # hidden state
        gates_pre_Left          = intermediate_results['gates_pre_Left']                    # load inter calc results
        gates_pre_Right         = intermediate_results['gates_pre_Right']
        gates_Left              = intermediate_results['gates_Left']
        gates_Right             = intermediate_results['gates_Right']

        Wxh_Left, Whh_Left, bxh_Left, Wxh_Right, Whh_Right, bxh_Right = weights             # split weights
        Wxh_Left, Whh_Left, bxh_Left    = Wxh_Left.T, Whh_Left.T, bxh_Left.T
        Wxh_Right, Whh_Right, bxh_Right = Wxh_Right.T, Whh_Right.T, bxh_Right.T
        T                               = inputs.shape[0]                               # time_step
        d                               = int(Wxh_Left.shape[0]/3)                          # units length
        input_col                       = inputs.shape[1]                               # every time_step, the input's diem, for bi-gru_1, equals to embedding diem
        inputs_rev                      = inputs[::-1, :].copy()
        idx_z, idx_r, idx_h             = np.arange(0,d), np.arange(d,2*d), np.arange(2*d,3*d)

        # initialize
        Rx              = np.zeros(inputs.shape)
        Rx_rev          = np.zeros(inputs.shape)

        Rh_Left         = np.zeros((T+1, d))
        Rc_Left         = np.zeros((T+1, d))
        Rg_Left         = np.zeros((T,   d)) # gate g only
        Rh_Right        = np.zeros((T+1, d))
        Rc_Right        = np.zeros((T+1, d))
        Rg_Right        = np.zeros((T,   d)) # gate g only

        if 'return_sequences' in kwargs.keys() and kwargs['return_sequences'] == True:
            Rh_Left[:T]         = R[:,:R.shape[1]//2]     # assign every unit with relevance, Rh should be normal order
            Rh_Right[:T]        = R[:,R.shape[1]//2:]

            for t in reversed(range(T)):            # T-1 -> 0
                Rh_Left[t-1]  += lrp_linear(gates_Left[t,idx_z]*h_Left[t-1], np.identity(d), np.zeros((d)), h_Left[t], Rh_Left[t], 2*d, eps, bias_factor)
                Rg_Left[t]    = lrp_linear((1-gates_Left[t,idx_z])*gates_Left[t,idx_h], np.identity(d), np.zeros((d)), h_Left[t], Rh_Left[t], 2*d, eps, bias_factor)
                Rx[t]         = lrp_linear(inputs[t], Wxh_Left[idx_h].T, bxh_Left[idx_h], gates_pre_Left[t, idx_h], Rg_Left[t], d + input_col, eps, bias_factor)
                Rh_Left[t-1]  += lrp_linear(h_Left[t-1] * gates_Left[t,idx_r], Whh_Left[idx_h].T, bxh_Left[idx_h], gates_pre_Left[t,idx_h], Rg_Left[t], d+input_col, eps, bias_factor)

                Rh_Right[t-1] += lrp_linear(gates_Right[t,idx_z]*h_Right[t-1], np.identity(d), np.zeros((d)), h_Right[t], Rh_Right[t], 2*d, eps, bias_factor)
                Rg_Right[t]   = lrp_linear((1-gates_Right[t,idx_z])*gates_Right[t,idx_h], np.identity(d), np.zeros((d)), h_Right[t], Rh_Right[t], 2*d, eps, bias_factor)
                Rx_rev[t]     = lrp_linear(inputs_rev[t], Wxh_Right[idx_h].T, bxh_Right[idx_h], gates_pre_Right[t,idx_h], Rg_Right[t], d+input_col, eps, bias_factor)
                Rh_Right[t-1] += lrp_linear(h_Right[t-1] * gates_Right[t,idx_r], Whh_Right[idx_h].T, bxh_Right[idx_h], gates_pre_Right[t,idx_h], Rg_Right[t], d+input_col, eps, bias_factor)


        else:
            Rh_Left[T-1]    = R[:len(R)//2]       # assign last 2 units with relevance
            Rh_Right[T-1]   = R[len(R)//2:]

            for t in reversed(range(T)):
                Rh_Left[t-1]  = lrp_linear(gates_Left[t,idx_z]*h_Left[t-1], np.identity(d), np.zeros((d)), h_Left[t], Rh_Left[t], 2*d, eps, bias_factor)
                Rg_Left[t]    = lrp_linear((1-gates_Left[t,idx_z])*gates_Left[t,idx_h], np.identity(d), np.zeros((d)), h_Left[t], Rh_Left[t], 2*d, eps, bias_factor)
                Rx[t]         = lrp_linear(inputs[t], Wxh_Left[idx_h].T, bxh_Left[idx_h], gates_pre_Left[t, idx_h], Rg_Left[t], d + input_col, eps, bias_factor)
                Rh_Left[t-1]  += lrp_linear(h_Left[t-1] * gates_Left[t,idx_r], Whh_Left[idx_h].T, bxh_Left[idx_h], gates_pre_Left[t,idx_h], Rg_Left[t], d+input_col, eps, bias_factor)

                Rh_Right[t-1] = lrp_linear(gates_Right[t,idx_z]*h_Right[t-1], np.identity(d), np.zeros((d)), h_Right[t], Rh_Right[t], 2*d, eps, bias_factor)
                Rg_Right[t]   = lrp_linear((1-gates_Right[t,idx_z])*gates_Right[t,idx_h], np.identity(d), np.zeros((d)), h_Right[t], Rh_Right[t], 2*d, eps, bias_factor)
                Rx_rev[t]     = lrp_linear(inputs_rev[t], Wxh_Right[idx_h].T, bxh_Right[idx_h], gates_pre_Right[t,idx_h], Rg_Right[t], d+input_col, eps, bias_factor)
                Rh_Right[t-1] += lrp_linear(h_Right[t-1] * gates_Right[t,idx_r], Whh_Right[idx_h].T, bxh_Right[idx_h], gates_pre_Right[t,idx_h], Rg_Right[t], d+input_col, eps, bias_factor)


        relevance_sum = Rh_Left[-1].sum()+Rc_Left[-1].sum()+Rh_Right[-1].sum()+Rc_Right[-1].sum()
        return np.add(Rx,Rx_rev[::-1,:]), relevance_sum


    @staticmethod
    def deeplift(inputs, inputs_ref, weights, R, eps, bias_factor, **kwargs):
        """
        kwargs
            recurrent_activation : activation function for GRU,LSTM, default hard_sigmoid
            return_sequences: same as Keras' return_sequences if True, the Rh should be (T,2*d), else (2*d,)
        """

        # forward pass to get intermediate results
        intermediate_results    = Bi_GRU.forward(inputs, weights, **kwargs)[1]
        h                       = intermediate_results['hidden_states']
        h_Left, h_Right         = h[:,:h.shape[1]//2], h[:,h.shape[1]//2:]                  # hidden state
        gates_pre_Left          = intermediate_results['gates_pre_Left']                    # load inter calc results
        gates_pre_Right         = intermediate_results['gates_pre_Right']
        gates_Left              = intermediate_results['gates_Left']
        gates_Right             = intermediate_results['gates_Right']

        # forward pass for reference inputs
        intermediate_results_ref    = Bi_GRU.forward(inputs_ref, weights, **kwargs)[1]
        h_ref                       = intermediate_results_ref['hidden_states']
        h_Left_ref, h_Right_ref     = h_ref[:,:h_ref.shape[1]//2], h_ref[:,h_ref.shape[1]//2:]                  # hidden state
        gates_pre_Left_ref          = intermediate_results_ref['gates_pre_Left']                    # load inter calc results
        gates_pre_Right_ref         = intermediate_results_ref['gates_pre_Right']
        gates_Left_ref              = intermediate_results_ref['gates_Left']
        gates_Right_ref             = intermediate_results_ref['gates_Right']


        Wxh_Left, Whh_Left, bxh_Left, Wxh_Right, Whh_Right, bxh_Right = weights             # split weights
        Wxh_Left, Whh_Left, bxh_Left    = Wxh_Left.T, Whh_Left.T, bxh_Left.T
        Wxh_Right, Whh_Right, bxh_Right = Wxh_Right.T, Whh_Right.T, bxh_Right.T
        inputs_rev                      = inputs[::-1,:].copy()
        inputs_ref_rev                  = inputs_ref[::-1,:].copy()
        T                               = inputs.shape[0]                               # time_step
        d                               = int(Wxh_Left.shape[0]/3)                          # units length
        input_col                       = inputs.shape[1]                               # every time_step, the input's diem, for bi-gru_1, equals to embedding diem
        idx_z, idx_r, idx_h             = np.arange(0,d), np.arange(d,2*d), np.arange(2*d,3*d)

        # initialize
        Rx              = np.zeros(inputs.shape)
        Rx_rev          = np.zeros(inputs.shape)

        Rh_Left         = np.zeros((T+1, d))
        Rc_Left         = np.zeros((T+1, d))
        Rg_Left         = np.zeros((T,   d)) # gate g only
        Rh_Right        = np.zeros((T+1, d))
        Rc_Right        = np.zeros((T+1, d))
        Rg_Right        = np.zeros((T,   d)) # gate g only


        # process the calc procedure said in layers.DeepLift()
        # combine the calc procedure rather than calc before every lrp_liner
        gates_Left[:,idx_h]         -= gates_Left_ref[:,idx_h]
        h_Left                      -= h_Left_ref
        inputs                      -= inputs_ref
        gates_pre_Left[:,idx_h]     -= gates_pre_Left_ref[:,idx_h]

        gates_Right[:,idx_h]        -= gates_Right_ref[:,idx_h]
        h_Right                     -= h_Right_ref
        inputs_rev                  -= inputs_ref_rev
        gates_pre_Right[:,idx_h]    -= gates_pre_Right_ref[:,idx_h]


        if 'return_sequences' in kwargs.keys() and kwargs['return_sequences'] == True:
            Rh_Left[:T]         = R[:,:R.shape[1]//2]     # assign every unit with relevance, Rh should be normal order
            Rh_Right[:T]        = R[:,R.shape[1]//2:]

            for t in reversed(range(T)):            # T-1 -> 0
                # # process the calc procedure said in layers.DeepLift()
                # gates_Left[t,:]         -= gates_Left_ref[t,:]
                # h_Left[t-1]             -= h_Left_ref[t-1]
                # inputs[t]               -= inputs_ref[t]
                # gates_pre_Left[t,:]     -= gates_pre_Left_ref[t,:]
                #
                # gates_Right[t,:]        -= gates_Right_ref[t,:]
                # h_Right[t-1]            -= h_Right_ref[t-1]
                # inputs_rev[t]           -= inputs_ref_rev[t]
                # gates_pre_Right[t,:]    -= gates_pre_Right_ref[t,:]

                Rh_Left[t-1]  += lrp_linear(gates_Left[t,idx_z]*h_Left[t-1], np.identity(d), np.zeros((d)), h_Left[t], Rh_Left[t], 2*d, eps, bias_factor)
                Rg_Left[t]    = lrp_linear((1-gates_Left[t,idx_z])*gates_Left[t,idx_h], np.identity(d), np.zeros((d)), h_Left[t], Rh_Left[t], 2*d, eps, bias_factor)
                Rx[t]         = lrp_linear(inputs[t], Wxh_Left[idx_h].T, bxh_Left[idx_h], gates_pre_Left[t, idx_h], Rg_Left[t], d + input_col, eps, bias_factor)
                Rh_Left[t-1]  += lrp_linear(h_Left[t-1] * gates_Left[t,idx_r], Whh_Left[idx_h].T, bxh_Left[idx_h], gates_pre_Left[t,idx_h], Rg_Left[t], d+input_col, eps, bias_factor)


                Rh_Right[t-1] += lrp_linear(gates_Right[t,idx_z]*h_Right[t-1], np.identity(d), np.zeros((d)), h_Right[t], Rh_Right[t], 2*d, eps, bias_factor)
                Rg_Right[t]   = lrp_linear((1-gates_Right[t,idx_z])*gates_Right[t,idx_h], np.identity(d), np.zeros((d)), h_Right[t], Rh_Right[t], 2*d, eps, bias_factor)
                Rx_rev[t]     = lrp_linear(inputs_rev[t], Wxh_Right[idx_h].T, bxh_Right[idx_h], gates_pre_Right[t,idx_h], Rg_Right[t], d+input_col, eps, bias_factor)
                Rh_Right[t-1] += lrp_linear(h_Right[t-1] * gates_Right[t,idx_r], Whh_Right[idx_h].T, bxh_Right[idx_h], gates_pre_Right[t,idx_h], Rg_Right[t], d+input_col, eps, bias_factor)


        else:
            Rh_Left[T-1]    = R[:len(R)//2]       # assign last 2 units with relevance
            Rh_Right[T-1]   = R[len(R)//2:]

            for t in reversed(range(T)):

                Rh_Left[t-1]  = lrp_linear(gates_Left[t,idx_z]*h_Left[t-1], np.identity(d), np.zeros((d)), h_Left[t], Rh_Left[t], 2*d, eps, bias_factor)
                Rg_Left[t]    = lrp_linear((1-gates_Left[t,idx_z])*gates_Left[t,idx_h], np.identity(d), np.zeros((d)), h_Left[t], Rh_Left[t], 2*d, eps, bias_factor)
                Rx[t]         = lrp_linear(inputs[t], Wxh_Left[idx_h].T, bxh_Left[idx_h], gates_pre_Left[t, idx_h], Rg_Left[t], d + input_col, eps, bias_factor)
                Rh_Left[t-1]  += lrp_linear(h_Left[t-1] * gates_Left[t,idx_r], Whh_Left[idx_h].T, bxh_Left[idx_h], gates_pre_Left[t,idx_h], Rg_Left[t], d+input_col, eps, bias_factor)


                Rh_Right[t-1] = lrp_linear(gates_Right[t,idx_z]*h_Right[t-1], np.identity(d), np.zeros((d)), h_Right[t], Rh_Right[t], 2*d, eps, bias_factor)
                Rg_Right[t]   = lrp_linear((1-gates_Right[t,idx_z])*gates_Right[t,idx_h], np.identity(d), np.zeros((d)), h_Right[t], Rh_Right[t], 2*d, eps, bias_factor)
                Rx_rev[t]     = lrp_linear(inputs_rev[t], Wxh_Right[idx_h].T, bxh_Right[idx_h], gates_pre_Right[t,idx_h], Rg_Right[t], d+input_col, eps, bias_factor)
                Rh_Right[t-1] += lrp_linear(h_Right[t-1] * gates_Right[t,idx_r], Whh_Right[idx_h].T, bxh_Right[idx_h], gates_pre_Right[t,idx_h], Rg_Right[t], d+input_col, eps, bias_factor)


        relevance_sum = Rh_Left[-1].sum()+Rc_Left[-1].sum()+Rh_Right[-1].sum()+Rc_Right[-1].sum()
        return np.add(Rx,Rx_rev[::-1,:]), relevance_sum




class Bi_LSTM(layers):

    def __init__(self):
        super(Bi_LSTM, self).__init__()


    @staticmethod
    def forward(inputs, weights, **kwargs):
        """
        kwargs:
            recurrent_activation : activation function for GRU,LSTM, default hard_sigmoid
            return_sequences: same as keras' return_sequences
        """

        # activation
        def hard_sigmoid(vec):
            zeros = np.zeros_like(vec)
            ones = zeros + 1
            return np.maximum(zeros, np.minimum(ones, 0.2 * vec + 0.5))

        if 'recurrent_activation' in kwargs and kwargs['recurrent_activation'] != 'hard_sigmoid':       # Todo*************************
            recurrent_activation = kwargs['recurrent_activation']
        else:
            recurrent_activation = hard_sigmoid

        Wxh_Left, Whh_Left, bxh_Left, Wxh_Right, Whh_Right, bxh_Right = weights                                     # split weights
        Wxh_Left, Whh_Left, bxh_Left    = Wxh_Left.T, Whh_Left.T, bxh_Left.T
        Wxh_Right, Whh_Right, bxh_Right = Wxh_Right.T, Whh_Right.T, bxh_Right.T
        T                          = inputs.shape[0]                                                                # time step
        d                          = int(Wxh_Left.shape[0] / 4)                                                     # units length
        idx                        = np.hstack((np.arange(0, 2*d), np.arange(3*d, 4*d))).astype(int)                # indices of the gates i,f,o ;
        idx_i, idx_f, idx_g, idx_o = np.arange(0,d), np.arange(d,2*d), np.arange(2*d,3*d), np.arange(3*d,4*d)       # weights order i,f,g,o

        # initialize
        gates_xh_Left   = np.zeros((T,4*d))
        gates_hh_Left   = np.zeros((T,4*d))
        gates_pre_Left  = np.zeros((T,4*d))  # gates signal pre-activation  needed by LRP
        gates_Left      = np.zeros((T,4*d))  # gates signal activation   needed by LRP

        gates_xh_Right  = np.zeros((T,4*d))
        gates_hh_Right  = np.zeros((T,4*d))
        gates_pre_Right = np.zeros((T,4*d))
        gates_Right     = np.zeros((T,4*d))

        h_Left          = np.zeros((T+1,d))  # hidden state h[-1] means h0
        h_Right         = np.zeros((T+1,d))
        c_Left          = np.zeros((T+1,d))  # memory signal c
        c_Right         = np.zeros((T+1,d))


        for t in range(T):
            gates_xh_Left[t]    = np.dot(Wxh_Left, inputs[t])                                                       # W*xt
            gates_hh_Left[t]    = np.dot(Whh_Left, h_Left[t-1])                                                     # U*ht-1
            gates_pre_Left[t]   = gates_xh_Left[t] + gates_hh_Left[t] + bxh_Left                                    # [W,U]*[xt,ht-1] + b
            gates_Left[t,idx]   = recurrent_activation(gates_pre_Left[t,idx])
            gates_Left[t,idx_g] = np.tanh(gates_pre_Left[t,idx_g])                                                  # gate signal i,f,g,o
            c_Left[t]           = gates_Left[t,idx_f]*c_Left[t-1] + gates_Left[t,idx_i]*gates_Left[t,idx_g]         # cell state c
            h_Left[t]           = gates_Left[t,idx_o]*np.tanh(c_Left[t])                                            # hidden state h

            inputs_rev          = inputs[::-1, :].copy()
            gates_xh_Right[t]   = np.dot(Wxh_Right, inputs_rev[t])
            gates_hh_Right[t]   = np.dot(Whh_Right, h_Right[t-1])
            gates_pre_Right[t]  = gates_xh_Right[t] + gates_hh_Right[t] + bxh_Right
            gates_Right[t,idx]  = recurrent_activation(gates_pre_Right[t,idx])
            gates_Right[t,idx_g]= np.tanh(gates_pre_Right[t,idx_g])                                                 # gate signal i,f,g,o
            c_Right[t]          = gates_Right[t,idx_f]*c_Right[t-1] + gates_Right[t,idx_i]*gates_Right[t, idx_g]
            h_Right[t]          = gates_Right[t,idx_o]*np.tanh(c_Right[t])


        intermediate_results = dict(hidden_states = np.concatenate((h_Left.copy(),h_Right.copy()), axis=1),
                                    gates_pre_Left=gates_pre_Left.copy(), gates_pre_Right=gates_pre_Right.copy(),
                                    gates_Left=gates_Left.copy(), gates_Right=gates_Right.copy(),
                                    c_Left=c_Left.copy(), c_Right=c_Right.copy())

        if 'return_sequences' in kwargs.keys() and kwargs['return_sequences'] == True:
            return np.concatenate((h_Left[:T].copy(), h_Right[:T][::-1,:].copy()), axis=1), intermediate_results
        else:
            return np.concatenate((h_Left[T-1],h_Right[T-1]), axis=0), intermediate_results


    @staticmethod
    def lrp(inputs, weights, R, eps, bias_factor, **kwargs):
        """
        kwargs
            LRP_class: if perform lrp for last layer, you must appoint one class's prediction scores to backward, default backward all neurons' scores
            return_sequences: if True, the Rh should be (T,2*d), else (2*d,)
        """

        intermediate_results    = Bi_LSTM.forward(inputs, weights, **kwargs)[1]    # forward pass, attention, this is for api sanity & easy, but the efficient could be low
        h                       = intermediate_results['hidden_states']
        h_Left, h_Right         = h[:,:h.shape[1]//2], h[:,h.shape[1]//2:]                      # hidden state
        # h_Right                 = h_Right[::-1,:] if 'return_sequences' in kwargs and kwargs['return_sequences'] == True else ...
        gates_pre_Left          = intermediate_results['gates_pre_Left']            # load inter calc results
        gates_pre_Right         = intermediate_results['gates_pre_Right']
        gates_Left              = intermediate_results['gates_Left']
        gates_Right             = intermediate_results['gates_Right']
        c_Left                  = intermediate_results['c_Left']
        c_Right                 = intermediate_results['c_Right']


        Wxh_Left, Whh_Left, bxh_Left, Wxh_Right, Whh_Right, bxh_Right = weights     # split weights
        Wxh_Left, Whh_Left, bxh_Left    = Wxh_Left.T, Whh_Left.T, bxh_Left.T
        Wxh_Right, Whh_Right, bxh_Right = Wxh_Right.T, Whh_Right.T, bxh_Right.T
        T                               = inputs.shape[0]                           # time_step
        d                               = int(Wxh_Left.shape[0]/4)                  # units length
        input_col                       = inputs.shape[1]                           # every time_step, the input's diem, for bi-gru_1, equals to embedding diem

        idx_i, idx_f, idx_g, idx_o      = np.arange(0,d), np.arange(d,2*d), np.arange(2*d,3*d), np.arange(3*d,4*d)

        # initialize
        Rx              = np.zeros(inputs.shape)
        Rx_rev          = np.zeros(inputs.shape)

        Rh_Left         = np.zeros((T+1, d))
        Rc_Left         = np.zeros((T+1, d))
        Rg_Left         = np.zeros((T,   d)) # gate g only
        Rh_Right        = np.zeros((T+1, d))
        Rc_Right        = np.zeros((T+1, d))
        Rg_Right        = np.zeros((T,   d)) # gate g only


        if 'return_sequences' in kwargs.keys() and kwargs['return_sequences'] == True:
            Rh_Left[:T]         = R[:,:R.shape[1]//2]     # assign every unit with relevance, Rh should be normal order
            Rh_Right[:T]        = R[:,R.shape[1]//2:]

            for t in reversed(range(T)):       # T-1 -> 0
                Rc_Left[t]     += Rh_Left[t]
                Rc_Left[t-1]    = lrp_linear(gates_Left[t,idx_f]*c_Left[t-1],         np.identity(d), np.zeros((d)), c_Left[t], Rc_Left[t], 2*d, eps, bias_factor)
                Rg_Left[t]      = lrp_linear(gates_Left[t,idx_i]*gates_Left[t,idx_g], np.identity(d), np.zeros((d)), c_Left[t], Rc_Left[t], 2*d, eps, bias_factor)
                Rx[t]           = lrp_linear(inputs[t], Wxh_Left[idx_g].T, bxh_Left[idx_g], gates_pre_Left[t, idx_g], Rg_Left[t], d + input_col, eps, bias_factor)
                Rh_Left[t-1]   += lrp_linear(h_Left[t-1], Whh_Left[idx_g].T, bxh_Left[idx_g], gates_pre_Left[t, idx_g], Rg_Left[t], d+input_col, eps, bias_factor)

                lrp_inputs_rev  = inputs[::-1,:]
                Rc_Right[t]    += Rh_Right[t]
                Rc_Right[t-1]   = lrp_linear(gates_Right[t,idx_f]*c_Right[t-1],         np.identity(d), np.zeros((d)), c_Right[t], Rc_Right[t], 2*d, eps, bias_factor)
                Rg_Right[t]     = lrp_linear(gates_Right[t,idx_i]*gates_Right[t,idx_g], np.identity(d), np.zeros((d)), c_Right[t], Rc_Right[t], 2*d, eps, bias_factor)
                Rx_rev[t]       = lrp_linear(lrp_inputs_rev[t], Wxh_Right[idx_g].T, bxh_Right[idx_g], gates_pre_Right[t, idx_g], Rg_Right[t], d+input_col, eps, bias_factor)
                Rh_Right[t-1]  += lrp_linear(h_Right[t-1], Whh_Right[idx_g].T, bxh_Right[idx_g], gates_pre_Right[t, idx_g], Rg_Right[t], d+input_col, eps, bias_factor)


        else:
            Rh_Left[T-1]    = R[:len(R)//2]    # assign last units with relevance, Rh should be normal order
            Rh_Right[T-1]   = R[len(R)//2:]

            for t in reversed(range(T)):       # T-1 -> 0
                Rc_Left[t]   += Rh_Left[t]
                Rc_Left[t-1]  = lrp_linear(gates_Left[t,idx_f]*c_Left[t-1],         np.identity(d), np.zeros((d)), c_Left[t], Rc_Left[t], 2*d, eps, bias_factor)
                Rg_Left[t]    = lrp_linear(gates_Left[t,idx_i]*gates_Left[t,idx_g], np.identity(d), np.zeros((d)), c_Left[t], Rc_Left[t], 2*d, eps, bias_factor)
                Rx[t]         = lrp_linear(inputs[t], Wxh_Left[idx_g].T, bxh_Left[idx_g], gates_pre_Left[t, idx_g], Rg_Left[t], d + input_col, eps, bias_factor)
                Rh_Left[t-1]  = lrp_linear(h_Left[t-1], Whh_Left[idx_g].T, bxh_Left[idx_g], gates_pre_Left[t, idx_g], Rg_Left[t], d+input_col, eps, bias_factor)

                lrp_inputs_rev = inputs[::-1, :]
                Rc_Right[t]  += Rh_Right[t]
                Rc_Right[t-1] = lrp_linear(gates_Right[t,idx_f]*c_Right[t-1],         np.identity(d), np.zeros((d)), c_Right[t], Rc_Right[t], 2*d, eps, bias_factor)
                Rg_Right[t]   = lrp_linear(gates_Right[t,idx_i]*gates_Right[t,idx_g], np.identity(d), np.zeros((d)), c_Right[t], Rc_Right[t], 2*d, eps, bias_factor)
                Rx_rev[t]     = lrp_linear(lrp_inputs_rev[t], Wxh_Right[idx_g].T, bxh_Right[idx_g], gates_pre_Right[t, idx_g], Rg_Right[t], d+input_col, eps, bias_factor)
                Rh_Right[t-1] = lrp_linear(h_Right[t-1], Whh_Right[idx_g].T, bxh_Right[idx_g], gates_pre_Right[t, idx_g], Rg_Right[t], d+input_col, eps, bias_factor)


        relevance_sum = Rh_Left[-1].sum()+Rc_Left[-1].sum()+Rh_Right[-1].sum()+Rc_Right[-1].sum()
        return np.add(Rx,Rx_rev[::-1,:]), relevance_sum



    @staticmethod
    def deeplift(inputs, inputs_ref, weights, R, eps, bias_factor, **kwargs):
        """
        kwargs
            recurrent_activation : activation function for GRU,LSTM, default hard_sigmoid
            return_sequences: same as Keras' return_sequences if True, the Rh should be (T,2*d), else (2*d,)
        """

        # forward pass to get intermediate results
        intermediate_results    = Bi_LSTM.forward(inputs, weights, **kwargs)[1]
        h                       = intermediate_results['hidden_states']
        h_Left, h_Right         = h[:,:h.shape[1]//2], h[:,h.shape[1]//2:]                  # hidden state
        gates_pre_Left          = intermediate_results['gates_pre_Left']                    # load inter calc results
        gates_pre_Right         = intermediate_results['gates_pre_Right']
        gates_Left              = intermediate_results['gates_Left']
        gates_Right             = intermediate_results['gates_Right']
        c_Left                  = intermediate_results['c_Left']
        c_Right                 = intermediate_results['c_Right']

        # forward pass for reference inputs
        intermediate_results_ref    = Bi_LSTM.forward(inputs_ref, weights, **kwargs)[1]
        h_ref                       = intermediate_results_ref['hidden_states']
        h_Left_ref, h_Right_ref     = h_ref[:,:h_ref.shape[1]//2], h_ref[:,h_ref.shape[1]//2:]                  # hidden state
        gates_pre_Left_ref          = intermediate_results_ref['gates_pre_Left']                    # load inter calc results
        gates_pre_Right_ref         = intermediate_results_ref['gates_pre_Right']
        gates_Left_ref              = intermediate_results_ref['gates_Left']
        gates_Right_ref             = intermediate_results_ref['gates_Right']
        c_Left_ref                  = intermediate_results_ref['c_Left']
        c_Right_ref                 = intermediate_results_ref['c_Right']


        Wxh_Left, Whh_Left, bxh_Left, Wxh_Right, Whh_Right, bxh_Right = weights             # split weights
        Wxh_Left, Whh_Left, bxh_Left    = Wxh_Left.T, Whh_Left.T, bxh_Left.T
        Wxh_Right, Whh_Right, bxh_Right = Wxh_Right.T, Whh_Right.T, bxh_Right.T
        inputs_rev                      = inputs[::-1,:].copy()
        inputs_ref_rev                  = inputs_ref[::-1,:].copy()
        T                               = inputs.shape[0]                               # time_step
        d                               = int(Wxh_Left.shape[0]/4)                          # units length
        input_col                       = inputs.shape[1]                               # every time_step, the input's diem, for bi-gru_1, equals to embedding diem
        idx_i, idx_f, idx_g, idx_o      = np.arange(0,d), np.arange(d,2*d), np.arange(2*d,3*d), np.arange(3*d,4*d)

        # initialize
        Rx              = np.zeros(inputs.shape)
        Rx_rev          = np.zeros(inputs.shape)

        Rh_Left         = np.zeros((T+1, d))
        Rc_Left         = np.zeros((T+1, d))
        Rg_Left         = np.zeros((T,   d)) # gate g only
        Rh_Right        = np.zeros((T+1, d))
        Rc_Right        = np.zeros((T+1, d))
        Rg_Right        = np.zeros((T,   d)) # gate g only


        # process the calc procedure said in layers.DeepLift()
        # combine the calc procedure rather than calc before every lrp_liner
        c_Left                      -= c_Left_ref
        gates_Left[:,idx_g]         -= gates_Left_ref[:,idx_g]
        h_Left                      -= h_Left_ref
        inputs                      -= inputs_ref
        gates_pre_Left[:,idx_g]     -= gates_pre_Left_ref[:,idx_g]

        c_Right                     -= c_Right_ref
        gates_Right[:,idx_g]        -= gates_Right_ref[:,idx_g]
        h_Right                     -= h_Right_ref
        inputs_rev                  -= inputs_ref_rev
        gates_pre_Right[:,idx_g]    -= gates_pre_Right_ref[:,idx_g]


        if 'return_sequences' in kwargs.keys() and kwargs['return_sequences'] == True:
            Rh_Left[:T]         = R[:,:R.shape[1]//2]     # assign every unit with relevance, Rh should be normal order
            Rh_Right[:T]        = R[:,R.shape[1]//2:]

            for t in reversed(range(T)):       # T-1 -> 0
                Rc_Left[t]     += Rh_Left[t]
                Rc_Left[t-1]    = lrp_linear(gates_Left[t,idx_f]*c_Left[t-1],         np.identity(d), np.zeros((d)), c_Left[t], Rc_Left[t], 2*d, eps, bias_factor)
                Rg_Left[t]      = lrp_linear(gates_Left[t,idx_i]*gates_Left[t,idx_g], np.identity(d), np.zeros((d)), c_Left[t], Rc_Left[t], 2*d, eps, bias_factor)
                Rx[t]           = lrp_linear(inputs[t], Wxh_Left[idx_g].T, bxh_Left[idx_g], gates_pre_Left[t, idx_g], Rg_Left[t], d + input_col, eps, bias_factor)
                Rh_Left[t-1]   += lrp_linear(h_Left[t-1], Whh_Left[idx_g].T, bxh_Left[idx_g], gates_pre_Left[t, idx_g], Rg_Left[t], d+input_col, eps, bias_factor)

                lrp_inputs_rev  = inputs[::-1,:]
                Rc_Right[t]    += Rh_Right[t]
                Rc_Right[t-1]   = lrp_linear(gates_Right[t,idx_f]*c_Right[t-1],         np.identity(d), np.zeros((d)), c_Right[t], Rc_Right[t], 2*d, eps, bias_factor)
                Rg_Right[t]     = lrp_linear(gates_Right[t,idx_i]*gates_Right[t,idx_g], np.identity(d), np.zeros((d)), c_Right[t], Rc_Right[t], 2*d, eps, bias_factor)
                Rx_rev[t]       = lrp_linear(lrp_inputs_rev[t], Wxh_Right[idx_g].T, bxh_Right[idx_g], gates_pre_Right[t, idx_g], Rg_Right[t], d+input_col, eps, bias_factor)
                Rh_Right[t-1]  += lrp_linear(h_Right[t-1], Whh_Right[idx_g].T, bxh_Right[idx_g], gates_pre_Right[t, idx_g], Rg_Right[t], d+input_col, eps, bias_factor)


        else:
            Rh_Left[T-1]    = R[:len(R)//2]    # assign last units with relevance, Rh should be normal order
            Rh_Right[T-1]   = R[len(R)//2:]

            for t in reversed(range(T)):       # T-1 -> 0
                Rc_Left[t]   += Rh_Left[t]
                Rc_Left[t-1]  = lrp_linear(gates_Left[t,idx_f]*c_Left[t-1],         np.identity(d), np.zeros((d)), c_Left[t], Rc_Left[t], 2*d, eps, bias_factor)
                Rg_Left[t]    = lrp_linear(gates_Left[t,idx_i]*gates_Left[t,idx_g], np.identity(d), np.zeros((d)), c_Left[t], Rc_Left[t], 2*d, eps, bias_factor)
                Rx[t]         = lrp_linear(inputs[t], Wxh_Left[idx_g].T, bxh_Left[idx_g], gates_pre_Left[t, idx_g], Rg_Left[t], d + input_col, eps, bias_factor)
                Rh_Left[t-1]  = lrp_linear(h_Left[t-1], Whh_Left[idx_g].T, bxh_Left[idx_g], gates_pre_Left[t, idx_g], Rg_Left[t], d+input_col, eps, bias_factor)

                lrp_inputs_rev = inputs[::-1, :]
                Rc_Right[t]  += Rh_Right[t]
                Rc_Right[t-1] = lrp_linear(gates_Right[t,idx_f]*c_Right[t-1],         np.identity(d), np.zeros((d)), c_Right[t], Rc_Right[t], 2*d, eps, bias_factor)
                Rg_Right[t]   = lrp_linear(gates_Right[t,idx_i]*gates_Right[t,idx_g], np.identity(d), np.zeros((d)), c_Right[t], Rc_Right[t], 2*d, eps, bias_factor)
                Rx_rev[t]     = lrp_linear(lrp_inputs_rev[t], Wxh_Right[idx_g].T, bxh_Right[idx_g], gates_pre_Right[t, idx_g], Rg_Right[t], d+input_col, eps, bias_factor)
                Rh_Right[t-1] = lrp_linear(h_Right[t-1], Whh_Right[idx_g].T, bxh_Right[idx_g], gates_pre_Right[t, idx_g], Rg_Right[t], d+input_col, eps, bias_factor)


        relevance_sum = Rh_Left[-1].sum()+Rc_Left[-1].sum()+Rh_Right[-1].sum()+Rc_Right[-1].sum()
        return np.add(Rx,Rx_rev[::-1,:]), relevance_sum