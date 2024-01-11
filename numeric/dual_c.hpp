#ifndef DUAL_NUMBER_HPP
#define DUAL_NUMBER_HPP

# не работи това в момента

#include <cmath>

class DualNumber {
public:
    // Constructors
    DualNumber(double realPart, double dualPart) : a(realPart), b(dualPart) {}

    // Operator overloads
    DualNumber operator*(const DualNumber& num) const {
        return DualNumber(a * num.a, b * num.a + a * num.b);
    }

    DualNumber operator*(double num) const {
        return DualNumber(a * num, b * num);
    }

    friend DualNumber operator*(double num, const DualNumber& dualNum) {
        return dualNum * num;
    }

    DualNumber operator+(const DualNumber& num) const {
        return DualNumber(a + num.a, b + num.b);
    }

    DualNumber operator+(double num) const {
        return DualNumber(a + num, b);
    }

    friend DualNumber operator+(double num, const DualNumber& dualNum) {
        return dualNum + num;
    }

    DualNumber operator-(const DualNumber& num) const {
        return DualNumber(a - num.a, b - num.b);
    }

    DualNumber operator-(double num) const {
        return DualNumber(a - num, b);
    }

    friend DualNumber operator-(double num, const DualNumber& dualNum) {
        return DualNumber(num - dualNum.a, -dualNum.b);
    }

    DualNumber operator/(const DualNumber& num) const {
        return DualNumber(a / num.a, (b * num.a - a * num.b) / (num.a * num.a));
    }

    DualNumber operator/(double num) const {
        return DualNumber(a / num, b / num);
    }

    friend DualNumber operator/(double num, const DualNumber& dualNum) {
        return DualNumber(num / dualNum.a, -num * dualNum.b / (dualNum.a * dualNum.a));
    }

    DualNumber operator-() const {
        return DualNumber(-a, -b);
    }

    DualNumber operator^(double power) const {
        return DualNumber(std::pow(a, power), b * power * std::pow(a, power - 1));
    }

    // Trigonometric functions
    DualNumber sin() const {
        return DualNumber(std::sin(a), b * std::cos(a));
    }

    DualNumber cos() const {
        return DualNumber(std::cos(a), -b * std::sin(a));
    }

    DualNumber tan() const {
        return sin() / cos();
    }

    // Logarithmic and exponential functions
    DualNumber log() const {
        return DualNumber(std::log(a), b / a);
    }

    DualNumber exp() const {
        return DualNumber(std::exp(a), b * std::exp(a));
    }

private:
    double a; // Real part
    double b; // Dual part
};


#endif // DUAL_NUMBER_HPP