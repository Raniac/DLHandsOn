#include <iostream>
#include <vector>
#include "DataObject.hpp"
#include "Utilities.hpp"

static void testDataObject() {
    DLHandsOn::DataObject();
    return;
}

int main() {
    testDataObject();
    DLHandsOn::assert(false, "Something went wrong!");
    return 0;
}