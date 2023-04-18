#ifndef TEST_SEMANTIC_CLASSIFICATION_WITH_DEDUPLICATION_CC
#define TEST_SEMANTIC_CLASSIFICATION_WITH_DEDUPLICATION_CC

#include "PDBString.h"
#include "PDBMap.h"
#include "DataTypes.h"
#include "TensorBlockIndex.h"
#include "InterfaceFunctions.h"
#include "PDBClient.h"
#include "FFMatrixBlock.h"
#include "FFMatrixUtil.h"
#include "SimpleFF.h"
#include "FFMatrixBlockScanner.h"
#include "FFMatrixWriter.h"
#include "FFAggMatrix.h"
#include "FFTransposeMult.h"
#include "SemanticClassifier.h"
#include <cstddef>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cstring>
#include <ctime>
#include <chrono>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string>

using namespace pdb;

void create_weights_set(pdb::PDBClient & pdbClient, std::string weight_set_name, int numBlock_x, int block_x, int matrix_totalNumBlock_y,
		int sharedNumBlock_y, int block_y, int numSharedPages) {

    std::string errMsg;
    pdbClient.removeSet("image-classification", weight_set_name, errMsg);
    //create private set 
    pdbClient.createSet<FFMatrixBlock>("image-classification", weight_set_name, errMsg,
                     DEFAULT_PAGE_SIZE, weight_set_name, nullptr, nullptr, false);

    //load blocks to the private set 
    ff::loadMatrix(pdbClient, "image-classification", weight_set_name, numBlock_x*block_x, (matrix_totalNumBlock_y-sharedNumBlock_y)*block_y, block_x, block_y, false, false, errMsg);

    sleep(30);
    bool whetherToAddSharedSet = true;
    //add the metadata of shared pages to the private set 
     for (int i = 0; i < numSharedPages-1; i++) {
          pdbClient.addSharedPage("image-classification", weight_set_name, "FFMatrixBlock", "image-classification",
			  "shared_weights", "FFMatrixBlock", i, 0, i, whetherToAddSharedSet, 0, errMsg );
	  whetherToAddSharedSet = false;
     } 


}

void execute(pdb::PDBClient & pdbClient, std::string weight_set_name, std::string input_set_name) {

  auto begin = std::chrono::high_resolution_clock::now();

  std::string errMsg;

  //create output set
  pdbClient.removeSet("image-classification", "outputs", errMsg);
  pdbClient.createSet<FFMatrixBlock>("image-classification", "outputs", errMsg,
                  DEFAULT_PAGE_SIZE, "outputs", nullptr, nullptr, false);


  // make the reader
  pdb::Handle<pdb::Computation> readA =
      makeObject<FFMatrixBlockScanner>("image-classification", weight_set_name);
  pdb::Handle<pdb::Computation> readB =
      makeObject<FFMatrixBlockScanner>("image-classification", input_set_name);

  // make the transpose multiply join
  pdb::Handle<pdb::Computation> join = pdb::makeObject<FFTransposeMult>();
  join->setInput(0, readA);
  join->setInput(1, readB);

  // make the transpose multiply aggregation
  pdb::Handle<pdb::Computation> myAggregation =
      pdb::makeObject<FFAggMatrix>();
  myAggregation->setInput(join);


  // make the classifier
  pdb::Handle<pdb::Computation> classifier = pdb::makeObject<SemanticClassifier>();
  classifier->setInput(myAggregation);

  // make the writer
  pdb::Handle<pdb::Computation> myWriter = nullptr;
  myWriter = pdb::makeObject<FFMatrixWriter>("image-classification", "outputs");
  myWriter->setInput(classifier);

  bool materializeHash = false;

  auto exe_begin = std::chrono::high_resolution_clock::now();

  // run the computation
  if (!pdbClient.executeComputations(errMsg, weight_set_name, materializeHash, myWriter)) {
      cout << "Computation failed. Message was: " << errMsg << "\n";
      exit(1);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "****Image Classification End-to-End Time Duration: ****"
              << std::chrono::duration_cast<std::chrono::duration<float>>(end - begin).count()
              << " secs." << std::endl;

  std::cout << "****Image Classification Execution Time Duration: ****"
              << std::chrono::duration_cast<std::chrono::duration<float>>(end - exe_begin).count()
              << " secs." << std::endl;

  //verify the results
  ff::print_stats(pdbClient, "image-classification", "outputs");
  ff::print(pdbClient, "image-classification", "outputs");

}


int main(int argc, char* argv[]) {

     bool loadData = true;
     if (argc > 1) {
        if (strcmp(argv[1], "N") == 0) {
           loadData = false;
	}
     }

     int numModels = 4;
     if (argc > 2) {
         numModels = atoi(argv[2]);
     }
     makeObjectAllocatorBlock(124 * 1024 * 1024, true);

    //create a shared set
     string masterIp = "localhost";
     pdb::PDBLoggerPtr clientLogger = make_shared<pdb::PDBLogger>("TestSharedSetLog");
     pdb::PDBClient pdbClient(8108, masterIp, clientLogger, false, true);
     pdb::CatalogClient catalogClient(8108, masterIp, clientLogger);

     std::string errMsg;

     int block_x = 100;
     int block_y = 512;

     int batchSize = 32;

     int matrix_totalNumBlock_y[4] = {5, 4, 3, 2};
     int numBlock_x[4] = {17, 11, 6, 1};

     if (loadData) {

         ff::createDatabase(pdbClient, "image-classification");
         ff::loadLibrary(pdbClient, "libraries/libFFMatrixMeta.so");
         ff::loadLibrary(pdbClient, "libraries/libFFMatrixData.so");
         ff::loadLibrary(pdbClient, "libraries/libFFMatrixBlock.so");
         ff::loadLibrary(pdbClient, "libraries/libFFMatrixBlockScanner.so");
         ff::loadLibrary(pdbClient, "libraries/libFFMatrixWriter.so");
         ff::loadLibrary(pdbClient, "libraries/libFFAggMatrix.so");
         ff::loadLibrary(pdbClient, "libraries/libFFTransposeMult.so");
         ff::loadLibrary(pdbClient, "libraries/libSemanticClassifier.so");

	    //create the set for storing shared weights 
        pdbClient.createSet<FFMatrixBlock>("image-classification", "shared_weights", errMsg,
                     DEFAULT_PAGE_SIZE, "weights", nullptr, nullptr, true);

        //load blocks to the shared set
        ff::loadMatrix(pdbClient, "image-classification", "shared_weights", numBlock_x[0]*block_x, matrix_totalNumBlock_y[0]*block_y, block_x, block_y, false, false, errMsg);

        for(int i=0; i < numModels; i++) {
            create_weights_set(pdbClient, "weights"+std::to_string(i), numBlock_x[i], block_x, matrix_totalNumBlock_y[i], matrix_totalNumBlock_y[i], block_y, 1);
        }

        for(int i=0; i < numModels; i++) {
            pdbClient.createSet<FFMatrixBlock>("image-classification", "inputs"+std::to_string(i), errMsg,
                     DEFAULT_PAGE_SIZE, "inputs", nullptr, nullptr, false);
            std::cout << "To load matrix for image-classification:inputs"+std::to_string(i) << std::endl;
            ff::loadMatrix(pdbClient, "image-classification", "inputs"+std::to_string(i), batchSize, matrix_totalNumBlock_y[i]*block_y, block_x,
                    block_y, false, false, errMsg);
        }
     }
   

     for (int i = 0; i < numModels; i++) {
         execute(pdbClient, "weights"+std::to_string(i), "inputs"+std::to_string(i));
     }
}

#endif
