#include <gtest/gtest.h>
#include "../include/mnist/library.h"

TEST(cpp_app_template, mul)
{
    GTEST_ASSERT_EQ(app::multiply(3.0, 4.0), 12.0);
}

TEST(cpp_app_template, hello)
{
    app::hello();
}
