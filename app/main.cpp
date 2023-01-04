#include <iostream>
#include <monogon/loss/MSE.h>
#include <monogon/optimizer/SGD.h>
#include <mnist_loader/mnist_reader.hpp>
#include <monogon/layer/Activation.h>
#include <monogon/layer/Dense.h>
#include <monogon/layer/Input.h>
#include <monogon/model/Model.h>
#include <monogon/tool/OneHot.h>

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

    OneHot oneHot;

    Array X = (x / 255.0);
    Array Y = oneHot(y, 10, 1.0, 0.0);

    ReLu relu;
    Sigmoid sigmoid;

    Input input({784});
    Dense dense1({128}, input);
    Activation activation1(relu, dense1);
    Dense dense2({10}, activation1);
    Activation activation2(sigmoid, dense2);

    Model model(dense1, activation2);
    model.compile(SGD(1.0), MSE());
    model.fit(X, Y, 5, 32);
    std::cout << model.predict(X);

    return 0;
}
