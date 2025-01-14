#ifndef OBJECTQUERYMODEL_DISTRIBUTEDSTORAGEMANAGERCLIENT_H
#define OBJECTQUERYMODEL_DISTRIBUTEDSTORAGEMANAGERCLIENT_H

#include "ServerFunctionality.h"
#include "PDBLogger.h"
#include "DataTypes.h"
#include "SimpleRequestResult.h"
#include "Computation.h"
#include "LambdaIdentifier.h"

namespace pdb {


// This functionality talks with the DistributedStorageManagerServer instance
// to execute storage-related user requests.


class DistributedStorageManagerClient : public ServerFunctionality {

public:
    DistributedStorageManagerClient();
    DistributedStorageManagerClient(int portIn, std::string addressIn, PDBLoggerPtr myLoggerIn);
    ~DistributedStorageManagerClient();

    void registerHandlers(PDBServer& forMe);  // no - op


    // TODO: Allow the user to specify what partitioning policy they want to use for this step in
    // the future
    bool createDatabase(const std::string& databaseName, std::string& errMsg);

    bool createSet(const std::string& databaseName,
                   const std::string& setName,
                   const std::string& typeName,
                   std::string& errMsg,
                   size_t pageSize = DEFAULT_PAGE_SIZE,
                   const std::string& createdJobId = "",
                   Handle<Computation> computationForDispatch = nullptr,
                   Handle<LambdaIdentifier> lambdaForDispatch = nullptr,
                   size_t desiredSize = 0,
                   bool isMRU = false,
		   bool isSharedTensorBlockSet = false,
		   bool isModelSet = false);


    bool createSet(const std::string& databaseName,
                   const std::string& setName,
                   const std::string& typeName,
                   std::string& errMsg,
                   size_t pageSize,
                   const std::string& createdJobId,
                   Handle<Vector<Handle<Computation>>> computationsForDispatch,
                   std::string jobName,
                   std::string jobName1,
                   std::string jobName2,
                   std::string computationName1,
                   std::string computationName2,
                   std::string lambdaName1,
                   std::string lambdaName2, 
                   size_t desiredSize = 0,
                   bool isMRU = false,
		   bool isSharedTensorBlockSet = false,
		   bool isModelSet = false);


    // create a temp set that only goes through storage
    bool createTempSet(const std::string& databaseName,
                       const std::string& setName,
                       const std::string& typeName,
                       std::string& errMsg,
                       size_t pageSize = DEFAULT_PAGE_SIZE,
                       const std::string& createdJobId = "",
                       Handle<Computation> computationForDispatch = nullptr,
                       Handle<LambdaIdentifier> lambdaForDispatch = nullptr,
                       size_t desiredSize = 0,
                       bool isMRU = false);

    // templated createSet
    template <class DataType>
    bool createSet(const std::string& databaseName,
                   const std::string& setName,
                   std::string& errMsg,
                   size_t pageSize = DEFAULT_PAGE_SIZE,
                   const std::string& createdJobId = "",
                   Handle<Computation> computationForDispatch = nullptr,
                   Handle<LambdaIdentifier> lambdaForDispatch = nullptr,
                   size_t desiredSize = 0,
                   bool isMRU = false,
		   bool isSharedTensorBlockSet = false,
		   bool isModelSet = false);


    // templated createSet
    template <class DataType>
    bool createSet(const std::string& databaseName,
                   const std::string& setName,
                   std::string& errMsg,
                   size_t pageSize,
                   const std::string& createdJobId,
                   Handle<Vector<Handle<Computation>>> computationsForDispatch,
                   std::string jobName,
                   std::string jobName1,
                   std::string jobName2,
                   std::string computationName1,
                   std::string computationName2,
                   std::string lambdaName1,
                   std::string lambdaName2,
                   size_t desiredSize = 0,
                   bool isMRU = false,
		   bool isSharedTensorBlockSet = false,
		   bool isModelSet = false);


   bool addSharedPage(std::string sharingDatabase,
                   std::string sharingSetName,
                   std::string sharingTypeName,
                   std::string sharedDatabase,
                   std::string sharedSetName,
                   std::string sharedTypeName,
                   PageID pageId,
                   FilePartitionID filePartitionId,
                   unsigned int pageSeqId,
                   bool whetherToAddSharedSet,
                   NodeID nodeId,
		   std::string& errMsg);


  /* Link a shared block to its actual position in a tensor*/
  bool addSharedMapping(std::string sharingDatabase,
                  std::string sharingSetName,
                  std::string sharingTypeName,
                  std::string sharedDatabase,
                  std::string sharedSetName,
                  std::string sharedTypeName,
                  std::string fileName,
                  size_t totalRows,
                  size_t totalCols,
		  bool transpose,
                  std::string& errMsg);




    // storage cleanup to flush buffered data to disk
    bool flushData(std::string& errMsg);

    bool removeDatabase(const std::string& databaseName, std::string& errMsg);

    bool removeSet(const std::string& databaseName,
                   const std::string& setName,
                   std::string& errMsg);

    bool removeHashSet(const std::string& hashSetName,
                       std::string& errMsg);


    // clean up all set by removing both in-memory and on-disk data
    bool clearSet(const std::string& databaseName,
                  const std::string& setName,
                  const std::string& typeName,
                  std::string& errMsg);

    // remove a temp set that only goes through storage
    bool removeTempSet(const std::string& databaseName,
                       const std::string& setName,
                       const std::string& typeName,
                       std::string& errMsg);

    // export set to files with user specified format
    // the exported set is partitioned over all storage nodes on the specified directory
    // Note that the objects in set must be instances of ExportableObject
    bool exportSet(const std::string& databaseName,
                   const std::string& setName,
                   const std::string& outputFilePath,
                   const std::string& format,
                   std::string& errMsg);


private:
    std::function<bool(Handle<SimpleRequestResult>)> generateResponseHandler(
        std::string description, std::string& errMsg);

    int port;
    std::string address;
    PDBLoggerPtr logger;
};
}

#include "DistributedStorageManagerClientTemplate.cc"
#endif  // OBJECTQUERYMODEL_DISTRIBUTEDSTORAGEMANAGERCLIENT_H
