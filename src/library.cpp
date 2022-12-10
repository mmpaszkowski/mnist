#include "../include/mnist/library.h"
#include <fmt/color.h>
#include <config.h>

namespace app
{
    void hello()
    {
        fmt::print(fg(fmt::color::white), project_name);
        fmt::print(fg(fmt::color::green), " v");
        fmt::print(fg(fmt::color::green), project_version);
        fmt::print(fg(fmt::color::green), "\n");
    }

    double multiply(double a, double b)
    {
        return a * b;
    }
}

