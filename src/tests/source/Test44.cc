
#ifndef TEST_44_H
#define TEST_44_H

#include "Join.h"
#include "PDBString.h"
#include "Query.h"
#include "Lambda.h"
#include "Selection.h"
#include "QueryClient.h"
#include "QueryOutput.h"
#include "StorageClient.h"
#include "ChrisSelection.h"
#include "StringSelection.h"
#include "SharedEmployee.h"
#include "QueryNodeIr.h"
#include "Selection.h"
#include "SelectionIr.h"
#include "Set.h"
#include "SourceSetNameIr.h"
#include "Supervisor.h"
#include "ProjectionIr.h"
#include "QueryGraphIr.h"
#include "QueryOutput.h"
#include "IrBuilder.h"
#include "QuerySchedulerServer.h"
#include "RecordPredicateIr.h"
#include "RecordProjectionIr.h"
#include "DataTypes.h"
#include <ctime>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <chrono>
#include <fcntl.h>

using namespace pdb;
using pdb_detail::QueryGraphIr;
using pdb_detail::ProjectionIr;
using pdb_detail::RecordPredicateIr;
using pdb_detail::SelectionIr;
using pdb_detail::SetExpressionIr;
using pdb_detail::SourceSetNameIr;
using pdb_detail::buildIr;


int main(int argc, char* argv[]) {

    auto begin = std::chrono::high_resolution_clock::now();

    bool printResult = true;
    bool clusterMode = false;
    if (argc > 1) {

        printResult = false;
        std::cout << "You successfully disabled printing result." << std::endl;

    } else {
        std::cout << "Will print result. If you don't want to print result, you can add any "
                     "character as the first parameter to disable result printing."
                  << std::endl;
    }

    if (argc > 2) {

        clusterMode = true;
        std::cout << "You successfully set the test to run on cluster." << std::endl;

    } else {
        std::cout << "Will run on local node. If you want to run on cluster, you can add any "
                     "character as the second parameter to run on the cluster configured by "
                     "$PDB_HOME/conf/serverlist."
                  << std::endl;
    }


    // for allocations
    const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

    // register this query class
    string errMsg;
    PDBLoggerPtr myLogger = make_shared<pdb::PDBLogger>("clientLog");
    ConfigurationPtr conf = make_shared<Configuration>();
    StorageClient temp(8108, "localhost", myLogger);

    temp.registerType("libraries/libChrisSelection.so", errMsg);
    temp.registerType("libraries/libStringSelection.so", errMsg);

    // connect to the query client
    QueryClient myClient(8108, "localhost", myLogger);

    PDBLoggerPtr logger = make_shared<PDBLogger>("client44.log");
    PDBServer fakeServerForScheduler(8109, 100, logger);  // port doesn't matter, will not listen


    // make the query graph
    Handle<Set<SharedEmployee>> myInputSet =
        myClient.getSet<SharedEmployee>("chris_db", "chris_set");
    Handle<ChrisSelection> myFirstSelect = makeObject<ChrisSelection>();
    myFirstSelect->setInput(myInputSet);
    Handle<QueryOutput<String>> outputOne =
        makeObject<QueryOutput<String>>("chris_db", "output_set1", myFirstSelect);
    // Handle <StringSelection> mySecondSelect = makeObject <StringSelection> ();
    // mySecondSelect->setInput (myFirstSelect);
    // Handle <QueryOutput <String>> outputTwo = makeObject <QueryOutput <String>> ("chris_db",
    // "output_set2", mySecondSelect);

    Handle<Vector<Handle<QueryBase>>> queries = makeObject<Vector<Handle<QueryBase>>>();
    queries->push_back(outputOne);
    // queries->push_back(outputTwo);


    // to build Ir
    // in our distributed PDB, buildIr is done in QuerySchedulerServer (Master node), instead of in
    // client
    // see Test49 for selection query against a distributed PDB
    pdb_detail::QueryGraphIrPtr queryGraph = buildIr(queries);

    // to initialize a QuerySchedulerServer instance and schedule the query for execution
    // in distributed PDB, this is contained in the Master node, and is transparent to client
    // see Test49 for selection query against a distributed PDB
    QuerySchedulerServer server(logger, conf);
    server.recordServer(fakeServerForScheduler);  // to enable worker queue for Scheduler
    server.parseOptimizedQuery(queryGraph);
    server.printCurrentPlan();
    if (clusterMode == false) {
        server.schedule("localhost", 8108, myLogger, Direct);
    } else {
        server.initialize(false);
        server.schedule();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time Duration: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << " ns."
              << std::endl;

    std::cout << std::endl;
    // print the resuts

    // collect and print the results
    int count = 0;
    if ((printResult == true) && (clusterMode == false)) {
        SetIterator<String> result = myClient.getSetIterator<String>("chris_db", "output_set1");
        std::cout << "Query results: \n";
        for (auto a : result) {
            std::cout << (*a) << "; ";
            count++;
        }
        std::cout << "\n";
        std::cout << "count=" << count << std::endl;
    } else if (clusterMode == false) {
        SetIterator<String> result = myClient.getSetIterator<String>("chris_db", "output_set1");
        for (auto a : result) {
            count++;
        }
        std::cout << "count=" << count << std::endl;
    }

    // delete output set
    if (clusterMode == false) {
        // and delete the sets
        myClient.deleteSet("chris_db", "output_set1");
    }
    int code = system("scripts/cleanupSoFiles.sh");
    if (code < 0) {
        std::cout << "Can't clean up so files" << std::endl;
    }
    return 0;
}

#endif
