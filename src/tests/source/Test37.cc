
#ifndef TEST_37_CC
#define TEST_37_CC

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

    // to register selection type
    temp.registerType("libraries/libChrisSelection.so", errMsg);
    temp.registerType("libraries/libStringSelection.so", errMsg);

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

    // for (int num = 0; num < 25; ++num) {
    // now, create a bunch of data
    void* storage = malloc(1024 * 1024 * 8);
    {
        pdb::makeObjectAllocatorBlock(storage, 1024 * 1024 * 8, true);
        pdb::Handle<pdb::Vector<pdb::Handle<SharedEmployee>>> storeMe =
            pdb::makeObject<pdb::Vector<pdb::Handle<SharedEmployee>>>();
        int i;
        try {

            for (i = 0; true; i++) {
                pdb::Handle<SharedEmployee> myData =
                    pdb::makeObject<SharedEmployee>("Joe Johnson" + to_string(i), i + 45);
                storeMe->push_back(myData);
            }

        } catch (pdb::NotEnoughSpace& n) {

            // we got here, so go ahead and store the vector
            if (!temp.storeData<SharedEmployee>(storeMe, "chris_db", "chris_set", errMsg)) {
                cout << "Not able to store data: " + errMsg;
                return 0;
            }
            std::cout << i << std::endl;
            std::cout << "stored the data!!\n";
        }
    }
    free(storage);
    //}

    // and shut down the server
    if (!temp.shutDownServer(errMsg))
        std::cout << "Shut down not clean: " << errMsg << "\n";
}

#endif
