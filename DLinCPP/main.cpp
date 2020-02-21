#include <iostream>
#include <vector>
#include "DataObject.h"
#include "Utilities.h"

static void testDataObject() {
    DLHandsOn::DataObject();
    return;
}

int main() {
    testDataObject();
    DLHandsOn::assert(false, "Something went wrong!");
    return 0;
}