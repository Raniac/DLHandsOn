#include <iostream>
#include <vector>
#include "DataObject.cpp"
#include "Utilities.cpp"

static void testDataObject() {
    DLHandsOn::DataObject();
    return;
}

int main() {
    testDataObject();
    DLHandsOn::assert(false, "Something went wrong!");
    return 0;
}