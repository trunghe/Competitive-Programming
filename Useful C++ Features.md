Source: https://www.quora.com/As-a-competitive-programmer-what-C++11-features-should-I-be-aware-of-and-what-are-some-examples-that-illustrate-their-usage

# The auto keyword
You can now get the compiler to infer types most of the time with auto:
```
auto i = 7;
```
# Standardized initialization with ```{}```
E.g.
```
int i{8};
string text{"hello"};
```
And this even works for instance variables. Best of all, it works for vectors:
```
vector<int> numbers{3, 4, 5, 6};
```
You can even implement this in your own constructors, methods and functions:
```
void print(initializer_list<string> texts) {
    for(auto value: texts) {
    cout << value << endl;
    }
}

print({"one", "two", "three", "four"});
```
(Here the loop is an enhanced ```for``` loop -- see next)
# Enhanced ```for``` loops;
These allow you to iterate over collections, arrays or even Strings without a loop index or an explicit iterator (although for classes, the iterator must be implemented):
```
int numbers[] = {1, 3, 4};
for(auto value: numbers) {
    cout << value << endl;
}
```
# Lambda expressions
These allow you to define anonymous functions which you can pass around. 
```
#include <iostream>
using namespace std;

void test(void (*pFunc)() ) {
    pFunc();
}

int main() {
    // func is a lambda expression
    auto func = [](){ cout << "Hello" << endl; };

    // Call func
    func();

    // Pass func to the test() function
    test(func);

    // Call the test function and pass it a lambda defined on the fly.
    test([](){ cout << "Hello again" << endl; });

    return 0;
}
```
Lambda expression can accept parameters:
```
auto pGreet = [](string name){ cout << "Hello " << name << endl; };

pGreet("Mike");
```
... and they can return values. Usually the return type can be inferred, but sometimes you need
to specify it using the new "trailing" return-type syntax:
```
auto pDivide = [](double a, double b) -> double {
    if(b == 0.0) {
        return 0;
    }
    return a/b;
};

cout << pDivide(10.0, 5.0) << endl;
```
You can also specify which local variables you want to use, or "capture" in your lambdas, and whether by value or reference:
```
#include <iostream>
using namespace std;

int main() {

    int one = 1;
    int two = 2;
    int three = 3;

    // Capture one and two by value
    // The round brackets on the end call the lambda expression immediately -- normally not something you'd want to do.
    [one, two]() {cout << one << ", " << two << endl;}();

    // Capture all local variables by value
    [=]() {cout << one << ", " << two << endl;}();

    // Default capture all local variables by value, but capture three by reference
    [=, &three]() {three=7; cout << one << ", " << two << endl;}();
    cout << three << endl;

    // Default capture all local variables by reference
    [&]() {three=7; two=8; cout << one << ", " << two << endl;}();
    cout << two << endl;

    // Default capture all local variables by reference, but two and three by value
    [&, two, three]() {one=100; cout << one << ", " << two << endl;}();
    cout << one << endl;

    return 0;
}
```
# RValue references
This require a bit of study, but basically they are references that can refer to "rvalues", or non-named temporary values:

Given a class called "Test" and defining a function which returns a Test (no longer a problem if you implement a move constructor):
```
Test getTest() {
    return Test();
}
```
... you can do the following. The double ampersand defines an rvalue reference (not a reference to a reference!)
```
Test &&rtest1 = getTest();
```
If you have overloaded functions, one which a "normal" lvalue reference and the other with an rvalue reference,
you can now tell which type of value, temporary or not, got passed to your function:
```
void check(const Test &value) {
    cout << "lValue function!" << endl;
}

void check(Test &&value) {
    cout << "rValue function!" << endl;
}
```
.. then the following
```
Test test1 = getTest();
check(test1);
check(getTest());
```
.. prints
```
lValue function!
rValue function!
```
# Move constructors
Rvalue references let you implement move constructors, which can grab the resources from temporary objects, which after all,
are not going to need them.
```
class Something {
    // Imagine that pBuffer is allocated memory in your constructors, and destroyed
    // in your destructor
    int *pBuffer;

    Something(Something &&other) {
        pBuffer = other.pBuffer;
        // Stop memory getting deleted by other's destructor.
        other.pBuffer = nullptr;
    }
}
```
Now if you do this, with the above getTest() function:
```
Test test1 = getTest();
```
... instead of Test being constructed, then copied with the copy constructor to the
temporary return value, then copied again to test1, its resources are only "moved" with the move constructor.

Normally this situation would get optimized anyway in C++ 98, but now you can be sure this code is not
innefficient.

Hopefully all this works ... some snippets I've just typed and haven't tested, others are taken
from my C++ course ....

Rvalue references and move constructors gave me the most pain to understand with C++ 11, but finally
I realised they are very simple --- it's just that it's a big topic if you get into it in a very serious and detailed way ....
