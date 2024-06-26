import numpy as np

def sigmoid(x, derivative=False):
    if derivative:
        return (np.exp(-x))/((np.exp(-x)+1)**2)
    return 1/(1 + np.exp(-x))

def softmax(x, derivative=False):
    # Numerically stable with large exponentials
    exps = np.exp(x - x.max())
    if derivative:
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    return exps / np.sum(exps, axis=0)




# number of nodes in each layer
input_layer=784
hidden_1=128
hidden_2=64
output_layer=10

# x_train = np.random.randn(input_layer)

x_train = np.array([
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.011764705882352941,
    0.07058823529411765,
    0.07058823529411765,
    0.07058823529411765,
    0.49411764705882355,
    0.5333333333333333,
    0.6862745098039216,
    0.10196078431372549,
    0.6509803921568628,
    1.0,
    0.9686274509803922,
    0.4980392156862745,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.11764705882352941,
    0.1411764705882353,
    0.3686274509803922,
    0.6039215686274509,
    0.6666666666666666,
    0.9921568627450981,
    0.9921568627450981,
    0.9921568627450981,
    0.9921568627450981,
    0.9921568627450981,
    0.8823529411764706,
    0.6745098039215687,
    0.9921568627450981,
    0.9490196078431372,
    0.7647058823529411,
    0.25098039215686274,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.19215686274509805,
    0.9333333333333333,
    0.9921568627450981,
    0.9921568627450981,
    0.9921568627450981,
    0.9921568627450981,
    0.9921568627450981,
    0.9921568627450981,
    0.9921568627450981,
    0.9921568627450981,
    0.984313725490196,
    0.36470588235294116,
    0.3215686274509804,
    0.3215686274509804,
    0.2196078431372549,
    0.15294117647058825,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.07058823529411765,
    0.8588235294117647,
    0.9921568627450981,
    0.9921568627450981,
    0.9921568627450981,
    0.9921568627450981,
    0.9921568627450981,
    0.7764705882352941,
    0.7137254901960784,
    0.9686274509803922,
    0.9450980392156862,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.3137254901960784,
    0.611764705882353,
    0.4196078431372549,
    0.9921568627450981,
    0.9921568627450981,
    0.803921568627451,
    0.043137254901960784,
    0.0,
    0.16862745098039217,
    0.6039215686274509,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.054901960784313725,
    0.00392156862745098,
    0.6039215686274509,
    0.9921568627450981,
    0.35294117647058826,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.5450980392156862,
    0.9921568627450981,
    0.7450980392156863,
    0.00784313725490196,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.043137254901960784,
    0.7450980392156863,
    0.9921568627450981,
    0.27450980392156865,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.13725490196078433,
    0.9450980392156862,
    0.8823529411764706,
    0.6274509803921569,
    0.4235294117647059,
    0.00392156862745098,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.3176470588235294,
    0.9411764705882353,
    0.9921568627450981,
    0.9921568627450981,
    0.4666666666666667,
    0.09803921568627451,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.17647058823529413,
    0.7294117647058823,
    0.9921568627450981,
    0.9921568627450981,
    0.5882352941176471,
    0.10588235294117647,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.06274509803921569,
    0.36470588235294116,
    0.9882352941176471,
    0.9921568627450981,
    0.7333333333333333,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.9764705882352941,
    0.9921568627450981,
    0.9764705882352941,
    0.25098039215686274,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.1803921568627451,
    0.5098039215686274,
    0.7176470588235294,
    0.9921568627450981,
    0.9921568627450981,
    0.8117647058823529,
    0.00784313725490196,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.15294117647058825,
    0.5803921568627451,
    0.8980392156862745,
    0.9921568627450981,
    0.9921568627450981,
    0.9921568627450981,
    0.9803921568627451,
    0.7137254901960784,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.09411764705882353,
    0.4470588235294118,
    0.8666666666666667,
    0.9921568627450981,
    0.9921568627450981,
    0.9921568627450981,
    0.9921568627450981,
    0.788235294117647,
    0.3058823529411765,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.09019607843137255,
    0.25882352941176473,
    0.8352941176470589,
    0.9921568627450981,
    0.9921568627450981,
    0.9921568627450981,
    0.9921568627450981,
    0.7764705882352941,
    0.3176470588235294,
    0.00784313725490196,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.07058823529411765,
    0.6705882352941176,
    0.8588235294117647,
    0.9921568627450981,
    0.9921568627450981,
    0.9921568627450981,
    0.9921568627450981,
    0.7647058823529411,
    0.3137254901960784,
    0.03529411764705882,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.21568627450980393,
    0.6745098039215687,
    0.8862745098039215,
    0.9921568627450981,
    0.9921568627450981,
    0.9921568627450981,
    0.9921568627450981,
    0.9568627450980393,
    0.5215686274509804,
    0.043137254901960784,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.5333333333333333,
    0.9921568627450981,
    0.9921568627450981,
    0.9921568627450981,
    0.8313725490196079,
    0.5294117647058824,
    0.5176470588235295,
    0.06274509803921569,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
])

params = {
    'W1':np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
    'W2':np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
    'W3':np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer)
}

# input layer activations becomes sample
params['A0'] = x_train

# input layer to hidden layer 1
params['Z1'] = np.dot(params["W1"], params['A0'])
params['A1'] = sigmoid(params['Z1'])

print(params['W1'])
print(params['A0'])
print(params['Z1'])
print(params['A1'])

# hidden layer 1 to hidden layer 2
params['Z2'] = np.dot(params["W2"], params['A1'])
params['A2'] = sigmoid(params['Z2'])

# hidden layer 2 to output layer
params['Z3'] = np.dot(params["W3"], params['A2'])
params['A3'] = softmax(params['Z3'])

output = params['A3']

y_train = np.random.randn(output_layer)

change_w = {}

print("params = {")
print("  W1:", params['W1'].shape)
print("  W2:", params['W2'].shape)
print("  W3:", params['W3'].shape)
print("")
print("  A0:", params['A0'].shape)
print("  A1:", params['A1'].shape)
print("  A2:", params['A2'].shape)
print("  A3:", params['A3'].shape)
print("")
print("  Z1:", params['Z1'].shape)
print("  Z2:", params['Z2'].shape)
print("  Z3:", params['Z3'].shape)
print("")
print("}")

# Calculate W3 update
error = 2 * (output - y_train) / output.shape[0] * softmax(params['Z3'], derivative=True)
print("error:", error.shape)
change_w['W3'] = np.outer(error, params['A2'])

# Calculate W2 update
error = np.dot(params['W3'].T, error) * sigmoid(params['Z2'], derivative=True)
print("error:", error.shape)
change_w['W2'] = np.outer(error, params['A1'])

# Calculate W1 update
error = np.dot(params['W2'].T, error) * sigmoid(params['Z1'], derivative=True)
print("error:", error.shape)
change_w['W1'] = np.outer(error, params['A0'])

print("change_w = {")
print("  W1:", change_w['W1'].shape)
print("  W2:", change_w['W2'].shape)
print("  W3:", change_w['W3'].shape)
print("")
print("}")
