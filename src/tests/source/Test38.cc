
#ifndef TEST_38_CC
#define TEST_38_CC

#include "StorageClient.h"
#include "PDBVector.h"
#include "InterfaceFunctions.h"
#include "ChrisSelection.h"
#include "StringSelection.h"

// this won't be visible to the v-table map, since it is not in the biult in types directory
#include "SharedEmployee.h"

int main() {

    std::cout << "Make sure to run bin/test601 or bin/test35 in a different window to provide a "
                 "catalog/storage server.\n";

    // register the shared employee class
    pdb::StorageClient temp(8108, "localhost", make_shared<pdb::PDBLogger>("clientLog"), true);

    string errMsg;
    if (!temp.registerType("libraries/libSharedEmployee.so", errMsg)) {
        cout << "Not able to register type: " + errMsg;
    } else {
        cout << "Registered type.\n";
    }

    // now, create a new database
    if (!temp.createDatabase("chris_db", errMsg)) {
        cout << "Not able to create database: " + errMsg;
    } else {
        cout << "Created database.\n";
    }

    // now, create a new set in that database
    if (!temp.createSet<SharedEmployee>("chris_db", "chris_set", errMsg)) {
        cout << "Not able to create set: " + errMsg;
    } else {
        cout << "Created set.\n";
    }

    // to register selection type
    temp.registerType("libraries/libChrisSelection.so", errMsg);
    temp.registerType("libraries/libStringSelection.so", errMsg);
}

#endif
