#include "monogon/tool/ConfusionMatrix.h"
#include <iostream>
#include <mnist_loader/mnist_reader.hpp>
#include <monogon/layer/Activation.h>
#include <monogon/layer/Dense.h>
#include <monogon/layer/Input.h>
#include <monogon/loss/MSE.h>
#include <monogon/model/Model.h>
#include <monogon/optimizer/SGD.h>
#include <monogon/tool/ArgMax.h>
#include <monogon/tool/OneHot.h>
#include <matplot/matplot.h>

int main()
{
    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    std::cout << "Number of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Number of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Number of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Number of test labels = " << dataset.test_labels.size() << std::endl;

    Array x(dataset.training_images);
    Array y(dataset.training_labels);

    Array x_test(dataset.test_images);
    Array y_test(dataset.test_labels);

    OneHot oneHot;

    Array X = (x / 255.0);
    Array Y = oneHot(y, 10, 1.0, 0.0);

    Array X_test = (x_test / 255.0);
    Array Y_test = oneHot(y_test, 10, 1.0, 0.0);

    ReLu relu;
    Sigmoid sigmoid;

    Input input({784});
    Dense dense1({128}, input);
    Activation activation1(relu, dense1);
    Dense dense2({10}, activation1);
    Activation activation2(sigmoid, dense2);

    Model model(dense1, activation2);
    model.compile(SGD(1.0), MSE());
    model.fit(X, Y, 100, 32);

    Array Y_pred = model.predict(X_test);

    ArgMax arg_max;
    Array Y_pred_arg_max = arg_max(Y_pred);
    Array Y_arg_max = arg_max(Y_test);

    CategoricalAccuracy categorical_accuracy;
    std::cout << "Test categorical accuracy: " << categorical_accuracy(Y_pred, Y_test) << std::endl;

    ConfusionMatrix confusion_matrix;
    Array confusion_result = confusion_matrix(Y_pred_arg_max, Y_arg_max);
    std::cout << "Confusion matrix: " << confusion_result << std::endl;

    to_vector_2d(confusion_result);
    matplot::heatmap(to_vector_2d(confusion_result));

    auto ax = matplot::gca();
    ax->x_axis().ticklabels({"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"});
    ax->y_axis().ticklabels({"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"});
    matplot::save("confusion_chart.svg");
    return 0;
}
