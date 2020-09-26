#define LIBRARY_SIZE size_t(200)*size_t(1024)*size_t(1024)
#include <chrono>
#include <cstddef>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include "BuiltInObjectTypeIDs.h"
#include "CatAddNodeToDatabaseRequest.h"
#include "CatAddNodeToSetRequest.h"
#include "CatCreateDatabaseRequest.h"
#include "CatCreateSetRequest.h"
#include "CatDeleteDatabaseRequest.h"
#include "CatDeleteSetRequest.h"
#include "CatRegisterType.h"
#include "CatRemoveNodeFromDatabaseRequest.h"
#include "CatRemoveNodeFromSetRequest.h"
#include "CatSetObjectTypeRequest.h"
#include "CatSharedLibraryByNameRequest.h"
#include "CatSharedLibraryRequest.h"
#include "CatTypeNameSearch.h"
#include "CatTypeNameSearchResult.h"
#include "CatTypeSearchResult.h"
#include "CatalogCloseSQLiteDBHandler.h"
#include "CatalogDatabaseMetadata.h"
#include "CatalogPrintMetadata.h"
#include "CatalogServer.h"
#include "SimpleRequestHandler.h"
#include "SimpleRequestResult.h"
#include "UseTemporaryAllocationBlock.h"

namespace pdb {

PDBCatalogPtr CatalogServer::getCatalog() { return pdbCatalog; }

int16_t CatalogServer::searchForObjectTypeName(std::string objectTypeName) {

  // first search for the type name in the vTable map (in case it is built in)
  if (VTableMap::lookupBuiltInType(objectTypeName) != -1)
    return VTableMap::lookupBuiltInType(objectTypeName);

  // if this type's vtable pointer has been previously extracted by this process
  // return the typeId
  if (allTypeNames.count(objectTypeName) != 0) {
    return allTypeNames[objectTypeName];
  }

  // otherwise fetch it from the Master Catalog Server only if this is a Worker
  // Node catalog
  if (this->isMasterCatalogServer == false) {
    return catalogClientConnectionToMasterCatalogServer.searchForObjectTypeName(
        objectTypeName);
  }
  // otherwise returns -1
  return -1;
}

// register handlers for processing requests to the Catalog Server
void CatalogServer::registerHandlers(PDBServer &forMe) {
  PDB_COUT << "Catalog Server registering handlers" << endl;

  // handles a request to register metadata of a new cluster Node in the catalog
  forMe.registerHandler(
      CatalogNodeMetadata_TYPEID,
      make_shared<SimpleRequestHandler<CatalogNodeMetadata>>([&](
          Handle<CatalogNodeMetadata> request, PDBCommunicatorPtr sendUsingMe) {

        std::string errMsg;

        PDB_COUT << "CatalogServer handler CatalogNodeMetadata_TYPEID calling "
                    "addNodeMetadata "
                 << string(request->getItemKey()) << endl;

        const UseTemporaryAllocationBlock block{1024 * 1024};

        // adds the node metadata
        bool res =
            getFunctionality<CatalogServer>().addNodeMetadata(request, errMsg);

        const UseTemporaryAllocationBlock tempBlock{1024};
        Handle<SimpleRequestResult> response =
            makeObject<SimpleRequestResult>(res, errMsg);

        // sends result to requester
        res = sendUsingMe->sendObject(response, errMsg);
        return make_pair(res, errMsg);
      }));

  // handles a request to register or update metadata for a Database in the
  // catalog
  forMe.registerHandler(
      CatalogDatabaseMetadata_TYPEID,
      make_shared<SimpleRequestHandler<CatalogDatabaseMetadata>>(
          [&](Handle<CatalogDatabaseMetadata> request,
              PDBCommunicatorPtr sendUsingMe) {

            std::string errMsg;

            PDB_COUT << "--->CatalogServer handler "
                        "CatalogDatabaseMetadata_TYPEID calling "
                        "addDatabaseMetadata"
                     << endl;
            const UseTemporaryAllocationBlock block{1024 * 1024};
            bool res = false;
            string itemKey = request->getItemKey().c_str();
            PDB_COUT << " Looking for key " + itemKey << endl;

            // if database doesn't exist inserts metadata, otherwise only
            // updates it
            if (isDatabaseRegistered(itemKey) == false) {
              res = getFunctionality<CatalogServer>().addDatabaseMetadata(
                  request, errMsg);
            } else {
              res = getFunctionality<CatalogServer>().updateDatabaseMetadata(
                  request, errMsg);
            }

            const UseTemporaryAllocationBlock tempBlock{1024};
            Handle<SimpleRequestResult> response =
                makeObject<SimpleRequestResult>(res, errMsg);

            // sends result to requester
            res = sendUsingMe->sendObject(response, errMsg);
            return make_pair(res, errMsg);
          }));

  // handles a request to register or update metadata of a Set in the catalog
  forMe.registerHandler(
      CatalogSetMetadata_TYPEID,
      make_shared<SimpleRequestHandler<CatalogSetMetadata>>([&](
          Handle<CatalogSetMetadata> request, PDBCommunicatorPtr sendUsingMe) {

        std::string errMsg;

        PDB_COUT << "--->CatalogServer handler CatalogSetMetadata_TYPEID "
                    "calling addSetMetadata"
                 << endl;
        const UseTemporaryAllocationBlock block{1024 * 1024};
        bool res =
            getFunctionality<CatalogServer>().addSetMetadata(request, errMsg);

        const UseTemporaryAllocationBlock tempBlock{1024};

        Handle<SimpleRequestResult> response =
            makeObject<SimpleRequestResult>(res, errMsg);

        // sends result to requester
        res = sendUsingMe->sendObject(response, errMsg);
        return make_pair(res, errMsg);
      }));

  // handles a request to display the contents of the Catalog that have changed
  // since a given
  // timestamp
  forMe.registerHandler(
      CatalogPrintMetadata_TYPEID,
      make_shared<SimpleRequestHandler<CatalogPrintMetadata>>([&](
          Handle<CatalogPrintMetadata> itemToPrint,
          PDBCommunicatorPtr sendUsingMe) {

        std::string errMsg;

        PDB_COUT << "--->Testing CatalogPrintMetadata handler with timeStamp "
                 << itemToPrint->getTimeStamp().c_str() << endl;

        getFunctionality<CatalogServer>().printCatalog(itemToPrint);

        // sends result to requester
        bool res = sendUsingMe->sendObject(itemToPrint, errMsg);
        return make_pair(res, errMsg);
      }));

  // handles a request to return the typeID of a Type given its name
  forMe.registerHandler(
      CatTypeNameSearch_TYPEID,
      make_shared<SimpleRequestHandler<CatTypeNameSearch>>([&](
          Handle<CatTypeNameSearch> request, PDBCommunicatorPtr sendUsingMe) {

        PDB_COUT << "received CatTypeNameSearch message" << endl;

        const LockGuard guard{workingMutex};
        const UseTemporaryAllocationBlock block{1024 * 1024};

        // ask the catalog server for the type ID given the type name
        int16_t typeID =
            getFunctionality<CatalogServer>().searchForObjectTypeName(
                request->getObjectTypeName());

        PDB_COUT << "searchForObjectTypeName for " +
                        request->getObjectTypeName() + " is " +
                        std::to_string(typeID)
                 << endl;

        const UseTemporaryAllocationBlock tempBlock{1024};
        Handle<CatTypeSearchResult> response =
            makeObject<CatTypeSearchResult>(typeID);

        // sends result to requester
        std::string errMsg;
        bool res = sendUsingMe->sendObject(response, errMsg);
        return make_pair(res, errMsg);
      }));

  // handles a request to retrieve an .so library given a Type Id
  forMe.registerHandler(
      CatSharedLibraryRequest_TYPEID,
      make_shared<SimpleRequestHandler<CatSharedLibraryRequest>>(
          [&](Handle<CatSharedLibraryRequest> request,
              PDBCommunicatorPtr sendUsingMe) {

            const LockGuard guard{workingMutex};
            const UseTemporaryAllocationBlock block{1024 * 1024};

            vector<char> *putResultHere = new vector<char>();
            std::string errMsg;
            int16_t typeID = request->getTypeID();

            PDB_COUT << "CatalogServer to handle CatSharedLibraryRequest to "
                        "get shared library "
                        "for typeID="
                     << std::to_string(typeID) << endl;

            // retrieves the .so library
            bool res = getFunctionality<CatalogServer>().getSharedLibrary(
                typeID, (*putResultHere), errMsg);

            if (!res) {
              const UseTemporaryAllocationBlock tempBlock{1024};
              Handle<Vector<char>> response = makeObject<Vector<char>>();

              // sends the response in case of failure
              res = sendUsingMe->sendObject(response, errMsg);
            } else {
              PDB_COUT << "On Catalog Server bytes returned " +
                              std::to_string((*putResultHere).size())
                       << endl;

              // allocates memory for the .so library bytes
              const UseTemporaryAllocationBlock temp{1024 +
                                                     (*putResultHere).size()};
              Handle<Vector<char>> response = makeObject<Vector<char>>(
                  (*putResultHere).size(), (*putResultHere).size());
              memmove(response->c_ptr(), (*putResultHere).data(),
                      (*putResultHere).size());

              // sends result to requester
              res = sendUsingMe->sendObject(response, errMsg);
            }

            delete putResultHere;

            // return the result
            return make_pair(res, errMsg);
          }));

  // handles a request to retrieve an .so library given a Type Name along with
  // its metadata (stored as a serialized CatalogUserTypeMetadata object)
  forMe.registerHandler(
      CatSharedLibraryByNameRequest_TYPEID,
      make_shared<SimpleRequestHandler<CatSharedLibraryByNameRequest>>([&](
          Handle<CatSharedLibraryByNameRequest> request,
          PDBCommunicatorPtr sendUsingMe) {

        auto begin = std::chrono::high_resolution_clock::now();
        auto getShared = begin;
        auto setLib = begin;
        auto addObject = begin;

        string typeName = request->getTypeLibraryName();
        int16_t typeId = 0;
        // if typeName is empty we are searching by typeId, hence retrieves the
        // typeId from the request object
        if (typeName.empty() == true) {
          typeId = request->getTypeLibraryId();
        } else {
          // if a typeName is provided, we are searching by that name, so first
          // we have to retrieve the typeID, given the provided typeName
          typeId = allTypeNames[typeName];
        }

        const LockGuard guard{workingMutex};
        PDB_COUT << "Triggering Handler CatalogServer "
                    "CatSharedLibraryByNameRequest for typeName="
                 << typeName << " and typeId=" << std::to_string(typeId)
                 << endl;

        bool res = false;
        string returnedBytes;
        std::string errMsg;

        PDB_COUT << std::string("CatalogServer to handle "
                                "CatSharedLibraryByNameRequest to get shared "
                                "library for typeName=")
                 << typeName << std::string(" and typeId=")
                 << std::to_string(typeId) << endl;

        // if this is the Master catalog retrieves .so bytes from local catalog
        // copy
        if (this->isMasterCatalogServer == true) {
          // Allocates 150Mb for sending .so libraries
          const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 150};

          Handle<CatalogUserTypeMetadata> response =
              makeObject<CatalogUserTypeMetadata>();

          PDB_COUT << "    Invoking getSharedLibrary(typeName) from "
                      "CatalogServer Handler "
                      "b/c this is Master Catalog "
                   << endl;

          // if the type is not registered in the Master Catalog just return
          // with a typeID = -1
          if (allTypeCodes.count(typeId) == 0) {

            const UseTemporaryAllocationBlock tempBlock{1024};

            // Creates an empty Object just to send the response to caller
            Handle<CatalogUserTypeMetadata> notFoundResponse =
                makeObject<CatalogUserTypeMetadata>();
            String newItemID("-1");
            notFoundResponse->setObjectId(newItemID);

            res = sendUsingMe->sendObject(notFoundResponse, errMsg);

          } else {
            const UseTemporaryAllocationBlock tempBlock{LIBRARY_SIZE};
            // resolves typeName given the typeId
            typeName = allTypeCodes[typeId];
            PDB_COUT << "Resolved typeName " << typeName
                     << "  for typeId=" + std::to_string(typeId) << endl;

            // the type was found in the catalog, retrieves metadata and bytes
            // of the Shared Library
            res = getFunctionality<CatalogServer>().getSharedLibraryByTypeName(
                typeName, response, returnedBytes, errMsg);

            getShared = std::chrono::high_resolution_clock::now();

            PDB_COUT << "    Bytes returned YES isMaster: " +
                            std::to_string(returnedBytes.size())
                     << endl;
            PDB_COUT << "typeId=" + string(response->getObjectID()) << endl;
            PDB_COUT << "ItemName=" + string(response->getItemName()) << endl;
            PDB_COUT << "ItemKey=" + string(response->getItemKey()) << endl;

            // copies the bytes of the Shared Library to the object to be sent
            // back to the caller
            response->setLibraryBytes(returnedBytes);

            setLib = std::chrono::high_resolution_clock::now();

            PDB_COUT << "Object Id isMaster: " + string(response->getObjectID())
                     << " | " << string(response->getItemKey())
                     << " | " + string(response->getItemName()) << endl;

            if (!res) {
              res = sendUsingMe->sendObject(response, errMsg);
            } else {
              PDB_COUT << "     Sending metadata and bytes to caller!" << endl;
              // sends result to requester
              res = sendUsingMe->sendObject(response, errMsg);
            }
          }
        } else {
          // if this is not the Master catalog, retrieves .so bytes from the
          // remote master catalog
          if (allTypeCodes.count(typeId) == 0) {
            // process the case where the type is not registered in this local
            // catalog, allocates 150Mb for sending .so libraries
            const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 150};

            Handle<CatalogUserTypeMetadata> response =
                makeObject<CatalogUserTypeMetadata>();

            PDB_COUT << "    Connecting to the Remote Catalog Server via "
                        "Catalog Client"
                     << endl;
            PDB_COUT << "    Invoking "
                        "CatalogClient.getSharedLibraryByName(typeName) from "
                        "CatalogServer b/c this is Local Catalog "
                     << endl;

            // uses a dummyObjectFile since this is just making a remote call to
            // the Catalog Master Server and what matters is the returned bytes.
            string dummyObjectFile = string("temp.so");

            // retrieves from remote catalog the Shared Library bytes in
            // "returnedBytes" and metadata in the "response" object
            res = catalogClientConnectionToMasterCatalogServer
                      .getSharedLibraryByTypeName(typeId, typeName,
                                                  dummyObjectFile, response,
                                                  returnedBytes, errMsg);

            getShared = std::chrono::high_resolution_clock::now();

            PDB_COUT << "     Bytes returned NOT isMaster: "
                     << std::to_string(returnedBytes.size()) << endl;

            // if the library was successfully retrieved, go ahead and resolve
            // vtable fixing in the local catalog, given the library and
            // metadata retrieved from the remote Master Catalog
            if (res == true) {
              res = getFunctionality<CatalogServer>().addObjectType(
                  typeId, returnedBytes, errMsg);
            }

            addObject = std::chrono::high_resolution_clock::now();

            if (!res) {
              PDB_COUT << "     before sending response Vtable not fixed!!!!!!"
                       << endl;

              PDB_COUT << errMsg << endl;
              const UseTemporaryAllocationBlock tempBlock{1024};

              Handle<CatalogUserTypeMetadata> notFoundResponse =
                  makeObject<CatalogUserTypeMetadata>();
              String newItemID("-1");
              notFoundResponse->setObjectId(newItemID);

              res = sendUsingMe->sendObject(notFoundResponse, errMsg);

            } else {
              // if retrieval was successful prepare and send object to caller
              PDB_COUT << "     before sending response Vtable fixed!!!!"
                       << endl;

              const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 150};

              // prepares Object to be sent to caller
              Handle<CatalogUserTypeMetadata> objectToBeSent =
                  makeObject<CatalogUserTypeMetadata>();

              String _retBytes(returnedBytes);
              char objectIDCharArray[50];
              sprintf(objectIDCharArray, "%d", typeId);
              String newItemID(objectIDCharArray);
              objectToBeSent->setObjectId(newItemID);
              objectToBeSent->setLibraryBytes(_retBytes);
              String newTypeName(typeName);
              objectToBeSent->setItemName(newTypeName);
              objectToBeSent->setItemKey(newTypeName);

              // sends result to requester
              res = sendUsingMe->sendObject(objectToBeSent, errMsg);
            } // end retrieval fail/successful
          } else {
            // process the case where the type is already registered in this
            // local catalog
            // Allocates 150Mb for sending .so libraries
            const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 150};

            Handle<CatalogUserTypeMetadata> response =
                makeObject<CatalogUserTypeMetadata>();

            typeName = allTypeCodes[typeId];
            PDB_COUT << "Resolved typeName" << typeName
                     << " for typeId=" << std::to_string(typeId) << endl;

            // retrieves from local catalog the Shared Library bytes in
            // "returnedBytes" and metadata in the "response" object
            res = getFunctionality<CatalogServer>().getSharedLibraryByTypeName(
                typeName, response, returnedBytes, errMsg);

            getShared = std::chrono::high_resolution_clock::now();

            PDB_COUT << "    Bytes returned No isMaster: "
                     << std::to_string(returnedBytes.size()) << endl;

            response->setLibraryBytes(returnedBytes);

            setLib = std::chrono::high_resolution_clock::now();

            PDB_COUT << "Object Id isLocal: " << string(response->getObjectID())
                     << " | " << string(response->getItemKey()) << " | "
                     << string(response->getItemName()) << endl;

            if (!res) {
                // sends result to requester
              const UseTemporaryAllocationBlock tempBlock{1024};
              res = sendUsingMe->sendObject(response, errMsg);
            } else {
              PDB_COUT << "     Sending metadata and Shared Library to caller!"
                       << endl;
              res = sendUsingMe->sendObject(response, errMsg);
            }

          } // end "if" type was found in local catalog or not
        }   // end "if" is Master or Local catalog case

        auto end = std::chrono::high_resolution_clock::now();

        PDB_COUT
            << "Time Duration for getShared:\t "
            << std::to_string(
                   std::chrono::duration_cast<std::chrono::duration<float>>(
                       getShared - begin)
                       .count())
            << " secs." << endl;
        PDB_COUT
            << "Time Duration for setLib:\t "
            << std::to_string(
                   std::chrono::duration_cast<std::chrono::duration<float>>(
                       setLib - getShared)
                       .count()) +
                   " secs."
            << endl;
        PDB_COUT
            << "Time Duration for addObject:\t "
            << std::to_string(
                   std::chrono::duration_cast<std::chrono::duration<float>>(
                       addObject - getShared)
                       .count()) +
                   " secs."
            << endl;

        return make_pair(res, errMsg);
      }));

  // handles a request to retrieve the TypeId of a Type, if it's not registered
  // returns -1
  forMe.registerHandler(
      CatSetObjectTypeRequest_TYPEID,
      make_shared<SimpleRequestHandler<CatSetObjectTypeRequest>>(
          [&](Handle<CatSetObjectTypeRequest> request,
              PDBCommunicatorPtr sendUsingMe) {

            const LockGuard guard{workingMutex};
            const UseTemporaryAllocationBlock block{1024 * 1024};

            // ask the catalog server for the type ID and then the name of the
            // type
            int16_t typeID = getFunctionality<CatalogServer>().getObjectType(
                request->getDatabaseName(), request->getSetName());

            PDB_COUT << "typeID for Set with dbName="
                     << string(request->getDatabaseName())
                     << " and setName=" << string(request->getSetName())
                     << " is " << std::to_string(typeID) << endl;

            const UseTemporaryAllocationBlock tempBlock{1024};
            Handle<CatTypeNameSearchResult> response;
            if (typeID >= 0)
              response = makeObject<CatTypeNameSearchResult>(
                  searchForObjectTypeName(typeID), true, "success");
            else
              response = makeObject<CatTypeNameSearchResult>(
                  "", false, "could not find requested type");

            // sends result to requester
            std::string errMsg;
            bool res = sendUsingMe->sendObject(response, errMsg);
            return make_pair(res, errMsg);
          }));

  // handle a request to register metadata for a new Database in the catalog
  forMe.registerHandler(
      CatCreateDatabaseRequest_TYPEID,
      make_shared<SimpleRequestHandler<CatCreateDatabaseRequest>>(
          [&](Handle<CatCreateDatabaseRequest> request,
              PDBCommunicatorPtr sendUsingMe) {

            const LockGuard guard{workingMutex};
            const UseTemporaryAllocationBlock block{1024 * 1024};

            std::string errMsg;

            // invokes adding Database metadata to catalog
            bool res = getFunctionality<CatalogServer>().addDatabase(
                request->dbToCreate(), errMsg);

            const UseTemporaryAllocationBlock tempBlock{1024};
            Handle<SimpleRequestResult> response =
                makeObject<SimpleRequestResult>(res, errMsg);

            // sends result to requester
            res = sendUsingMe->sendObject(response, errMsg);
            return make_pair(res, errMsg);
          }));

  // handle a request to register metadata for a new Set in the catalog
  forMe.registerHandler(
      CatCreateSetRequest_TYPEID,
      make_shared<SimpleRequestHandler<CatCreateSetRequest>>([&](
          Handle<CatCreateSetRequest> request, PDBCommunicatorPtr sendUsingMe) {

        const LockGuard guard{workingMutex};
        const UseTemporaryAllocationBlock block{1024 * 1024};

        std::string errMsg;
        auto info = request->whichSet();

        // invokes adding Set metadata to catalog
        bool res = getFunctionality<CatalogServer>().addSet(
            request->whichType(), info.first, info.second, errMsg);

        const UseTemporaryAllocationBlock tempBlock{1024};
        Handle<SimpleRequestResult> response =
            makeObject<SimpleRequestResult>(res, errMsg);

        // sends result to requester
        res = sendUsingMe->sendObject(response, errMsg);
        return make_pair(res, errMsg);
      }));

  // handle a request to delete metadata for an existing Database in the catalog
  forMe.registerHandler(
      CatDeleteDatabaseRequest_TYPEID,
      make_shared<SimpleRequestHandler<CatDeleteDatabaseRequest>>(
          [&](Handle<CatDeleteDatabaseRequest> request,
              PDBCommunicatorPtr sendUsingMe) {

            const LockGuard guard{workingMutex};
            const UseTemporaryAllocationBlock block{1024 * 1024};

            std::string errMsg;

            // invokes deleting Database metadata from catalog
            bool res = getFunctionality<CatalogServer>().deleteDatabase(
                request->dbToDelete(), errMsg);

            const UseTemporaryAllocationBlock tempBlock{1024};
            Handle<SimpleRequestResult> response =
                makeObject<SimpleRequestResult>(res, errMsg);

            // sends result to requester
            res = sendUsingMe->sendObject(response, errMsg);
            return make_pair(res, errMsg);
          }));

  // handle a request to delete metadata for an existing Set in the catalog
  forMe.registerHandler(
      CatDeleteSetRequest_TYPEID,
      make_shared<SimpleRequestHandler<CatDeleteSetRequest>>([&](
          Handle<CatDeleteSetRequest> request, PDBCommunicatorPtr sendUsingMe) {

        const LockGuard guard{workingMutex};
        const UseTemporaryAllocationBlock block{1024 * 1024};

        std::string errMsg;
        auto info = request->whichSet();

        // invokes deleting Set metadata from catalog
        bool res = getFunctionality<CatalogServer>().deleteSet(
            info.first, info.second, errMsg);

        const UseTemporaryAllocationBlock tempBlock{1024};
        Handle<SimpleRequestResult> response =
            makeObject<SimpleRequestResult>(res, errMsg);

        // sends result to requester
        res = sendUsingMe->sendObject(response, errMsg);
        return make_pair(res, errMsg);
      }));

  // handle a request to add information of a node to an existing Database
  // this is invoked when the Distributed Storage Manager creates storage in
  // a node in the cluster for a given Database
  forMe.registerHandler(
      CatAddNodeToDatabaseRequest_TYPEID,
      make_shared<SimpleRequestHandler<CatAddNodeToDatabaseRequest>>(
          [&](Handle<CatAddNodeToDatabaseRequest> request,
              PDBCommunicatorPtr sendUsingMe) {

            const LockGuard guard{workingMutex};
            const UseTemporaryAllocationBlock block{1024 * 1024};

            std::string errMsg;

            // invokes adding node metadata to an Existing Database
            bool res = getFunctionality<CatalogServer>().addNodeToDB(
                request->nodeToAdd(), request->whichDB(), errMsg);

            const UseTemporaryAllocationBlock tempBlock{1024};

            Handle<SimpleRequestResult> response =
                makeObject<SimpleRequestResult>(res, errMsg);

            // sends result to requester
            res = sendUsingMe->sendObject(response, errMsg);
            return make_pair(res, errMsg);
          }));

  // handle a request to add information of a set to an existing Database
  // this is invoked when the Distributed Storage Manager creates a set
  // in a given node for an existing Database
  forMe.registerHandler(
      CatAddNodeToSetRequest_TYPEID,
      make_shared<SimpleRequestHandler<CatAddNodeToSetRequest>>(
          [&](Handle<CatAddNodeToSetRequest> request,
              PDBCommunicatorPtr sendUsingMe) {

            const LockGuard guard{workingMutex};
            const UseTemporaryAllocationBlock block{1024 * 1024};

            std::string errMsg;

            // invokes add Node information to a Set in a given Database
            bool res = getFunctionality<CatalogServer>().addNodeToSet(
                request->nodeToAdd(), request->whichDB(), request->whichSet(),
                errMsg);

            const UseTemporaryAllocationBlock tempBlock{1024};
            Handle<SimpleRequestResult> response =
                makeObject<SimpleRequestResult>(res, errMsg);

            // sends result to requester
            res = sendUsingMe->sendObject(response, errMsg);
            return make_pair(res, errMsg);
          }));

  // handle a request to remove information of a node to an existing Database
  // this is invoked when the Distributed Storage Manager deletes storage in
  // a node in the cluster for a given Database
  forMe.registerHandler(
      CatRemoveNodeFromDatabaseRequest_TYPEID,
      make_shared<SimpleRequestHandler<CatRemoveNodeFromDatabaseRequest>>(
          [&](Handle<CatRemoveNodeFromDatabaseRequest> request,
              PDBCommunicatorPtr sendUsingMe) {

            const LockGuard guard{workingMutex};
            const UseTemporaryAllocationBlock block{1024 * 1024};
            // ask the catalog server for the type ID and then the name of the
            // type
            std::string errMsg;

            // invokes remove node information from an existing Database
            bool res = getFunctionality<CatalogServer>().removeNodeFromDB(
                request->nodeToRemove(), request->whichDB(), errMsg);

            const UseTemporaryAllocationBlock tempBlock{1024};
            Handle<SimpleRequestResult> response =
                makeObject<SimpleRequestResult>(res, errMsg);

            // sends result to requester
            res = sendUsingMe->sendObject(response, errMsg);
            return make_pair(res, errMsg);
          }));

  // handle a request to remove information of a set to an existing Database
  // this is invoked when the Distributed Storage Manager deletes a set
  // in a given node for an existing Database
  forMe.registerHandler(
      CatRemoveNodeFromSetRequest_TYPEID,
      make_shared<SimpleRequestHandler<CatRemoveNodeFromSetRequest>>(
          [&](Handle<CatRemoveNodeFromSetRequest> request,
              PDBCommunicatorPtr sendUsingMe) {

            const LockGuard guard{workingMutex};
            const UseTemporaryAllocationBlock block{1024 * 1024};

            std::string errMsg;
            auto info = request->whichSet();

            // invokes remove node information from a Set in an existing
            // Database
            bool res = getFunctionality<CatalogServer>().removeNodeFromSet(
                request->nodeToRemove(), request->whichDB(),
                request->whichSet(), errMsg);

            const UseTemporaryAllocationBlock tempBlock{1024};
            Handle<SimpleRequestResult> response =
                makeObject<SimpleRequestResult>(res, errMsg);

            // sends result to requester
            res = sendUsingMe->sendObject(response, errMsg);
            return make_pair(res, errMsg);
          }));

  // handles a request to register a shared library
  forMe.registerHandler(
      CatRegisterType_TYPEID,
      make_shared<SimpleRequestHandler<CatRegisterType>>(
          [&](Handle<CatRegisterType> request, PDBCommunicatorPtr sendUsingMe) {
            PDB_COUT << "Got a CatRegisterType request" << std::endl;

            const LockGuard guard{workingMutex};
            PDB_COUT << "Got lockGuard" << std::endl;
            const UseTemporaryAllocationBlock block{1024 * 1024};

            // get the next object... this holds the shared library file... it
            // could be big, so be careful!!
            size_t objectSize = sendUsingMe->getSizeOfNextObject();
            PDB_COUT << "Got objectSize=" << objectSize << std::endl;
            bool res;
            std::string errMsg;
            void *memory = malloc(objectSize+2048);
            Handle<Vector<char>> myFile =
                sendUsingMe->getNextObject<Vector<char>>(memory, res, errMsg);
            PDB_COUT << "Received all data" << std::endl;
            if (res) {
              string soFile(myFile->c_ptr(), objectSize);
              PDB_COUT << "Create a large string on stack" << std::endl;
              res = (addObjectType(-1, soFile, errMsg) >= 0);
            }
            free(memory);

            const UseTemporaryAllocationBlock tempBlock{1024};
            Handle<SimpleRequestResult> response =
                makeObject<SimpleRequestResult>(res, errMsg);

            // sends result to requester
            res = sendUsingMe->sendObject(response, errMsg);
            return make_pair(res, errMsg);
          }));

  // handle to close the SQLite DB Handler
  forMe.registerHandler(
      CatalogCloseSQLiteDBHandler_TYPEID,
      make_shared<SimpleRequestHandler<CatalogCloseSQLiteDBHandler>>([&](
          Handle<CatalogCloseSQLiteDBHandler> request,
          PDBCommunicatorPtr sendUsingMe) {

        std::string errMsg;
        PDB_COUT << "--->Testing CatalogCloseSQLiteDBHandler handler " << endl;
        const UseTemporaryAllocationBlock block{1024 * 1024};

        pdbCatalog->closeSQLiteHandler();

        const UseTemporaryAllocationBlock tempBlock{1024};
        Handle<SimpleRequestResult> response =
            makeObject<SimpleRequestResult>(true, errMsg);

        // sends result to requester
        bool res = sendUsingMe->sendObject(response, errMsg);
        return make_pair(res, errMsg);
      }));
}

// returns the Name of a Type given its Type ID
std::string CatalogServer::searchForObjectTypeName(int16_t typeIdentifier) {

  PDB_COUT << "searchForObjectTypeName with typeIdentifier ="
           << std::to_string(typeIdentifier) << endl;

  // first search for the type name in the vTable map (in case it is built in)
  std::string result = VTableMap::lookupBuiltInType(typeIdentifier);
  if (result != "")
    return result;

  // return a -1 if we've never seen this type name
  if (allTypeCodes.count(typeIdentifier) == 0)
    return "";

  // return the name of a non built-in type (registered via Shared Library)
  return allTypeCodes[typeIdentifier];
}

// retrieves the bytes of a Shared Library and returns them in putResultHere
bool CatalogServer::getSharedLibrary(int16_t identifier,
                                     vector<char> &putResultHere,
                                     std::string &errMsg) {

  // first, make sure we have this identifier
  if (allTypeCodes.count(identifier) == 0) {
    errMsg = "CatalogServer::getSharedLibrary(): Error: didn't know the "
             "identifier you sent me";
    PDB_COUT << errMsg << endl;
    return false;
  }

  // now, read in the .so file
  std::string whichFile =
      catalogDirectory + "/tmp_so_files/" + allTypeCodes[identifier] + ".so";
  std::ifstream in(whichFile, std::ifstream::ate | std::ifstream::binary);
  size_t fileLen = in.tellg();

  struct stat st;
  stat(whichFile.c_str(), &st);
  fileLen = st.st_size;
  int filedesc = open(whichFile.c_str(), O_RDONLY);
  putResultHere.resize(fileLen);

  // put the bytes in the output parameter  "putResultHere"
  read(filedesc, putResultHere.data(), fileLen);
  close(filedesc);

  return true;
}

// retrieves the bytes of a Shared Library and its associated metadata, given
// its typeName
bool CatalogServer::getSharedLibraryByTypeName(
    std::string typeName, Handle<CatalogUserTypeMetadata> &itemMetadata,
    string &returnedBytes, std::string &errMsg) {

  PDB_COUT << " Catalog Server->inside get getSharedLibraryByName id for type "
           << typeName << endl;

  int metadataCategory = (int)PDBCatalogMsgType::CatalogPDBRegisteredObject;

  // retrieves the type id given its name
  string id = pdbCatalog->itemName2ItemId(metadataCategory, typeName);

  PDB_COUT << " id " << id << endl;

  // data_types is used for retrieving User Defined Types
  string typeOfObject = "data_types";

  if (pdbCatalog->retrievesDynamicLibrary(typeName, typeOfObject, itemMetadata,
                                          returnedBytes, errMsg) == true) {

    PDB_COUT << "Metadata returned at get SharedLibrary Id: "
             << string(itemMetadata->getItemId()) << endl;
    PDB_COUT << "Metadata returned at get SharedLibrary Key: "
             << string(itemMetadata->getItemKey()) << endl;
    PDB_COUT << "Bytes after string " << std::to_string(returnedBytes.size())
             << endl;
    return true;
  } else {
    PDB_COUT << "Item with key " << typeName << " was not found! " << endl;
    return false;
  }
}

// returns the typeId of a Type given it's name, if not found returns -1
int16_t CatalogServer::getObjectType(std::string databaseName,
                                     std::string setName) {
  if (setTypes.count(make_pair(databaseName, setName)) == 0)
    return -1;

  return setTypes[make_pair(databaseName, setName)];
}

// Adds metadata and bytes of a Shared Library in the catalog and returns its
// typeId
int16_t CatalogServer::addObjectType(int16_t typeIDFromMasterCatalog,
                                     string &soFile, string &errMsg) {

  // read the bytes from the temporary extracted file and copy to output
  // parameter
  string tempFile = catalogDirectory + "/tmp_so_files/temp.so";
  int filedesc = open(tempFile.c_str(), O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
  size_t sizeWritten = write(filedesc, soFile.data(), soFile.size());
  PDB_COUT << "addObjectType: sizeWritten=" << sizeWritten << std::endl;
  close(filedesc);

  // check to make sure it is valid Shared Library
  void *so_handle = nullptr;
  so_handle = dlopen(tempFile.c_str(), RTLD_LOCAL | RTLD_LAZY);
  if (!so_handle) {
    const char *dlsym_error = dlerror();
    errMsg = "Cannot process shared library. " + string(dlsym_error) + '\n';
    std::cout << "CatalogServer: " << errMsg << std::endl;
    dlclose(so_handle);
    return -1;
  }

  // makes a call to the "getObjectTypeName" function defined in the shared
  // library
  const char *dlsym_error;
  std::string getName = "getObjectTypeName";
  typedef char *getObjectTypeNameFunc();
  getObjectTypeNameFunc *myFunc =
      (getObjectTypeNameFunc *)dlsym(so_handle, getName.c_str());

  PDB_COUT << "open function: " << getName << endl;

  // if the function is not defined or there was an error return
  if ((dlsym_error = dlerror())) {
    errMsg =
        "Error, can't load function getObjectTypeName in the shared library. " +
        string(dlsym_error) + '\n';
    PDB_COUT << errMsg << endl;
    return -1;
  }
  PDB_COUT << "all ok" << endl;

  // now, get the type name and write the appropriate file
  string typeName(myFunc());
  dlclose(so_handle);

  PDB_COUT << "typeName returned from SO file: " << typeName << endl;

  // rename temporary file
  string newName = catalogDirectory + "/tmp_so_files/" + typeName + ".so";
  int result = rename(tempFile.c_str(), newName.c_str());
  if (result == 0) {
    PDB_COUT << "Successfully renaming file " << newName << endl;
  } else {
    PDB_COUT << "Renaming temp file failed " << newName << endl;
    return -1;
  }

  // add the new type name, if we don't already have it
  if (allTypeNames.count(typeName) == 0) {
    PDB_COUT << "Fixing vtable ptr for type " << typeName
             << " with metadata retrieved from remote Catalog Server." << endl;

    int16_t typeCode;

    // if the type received is -1 this is a type not registered and we set the
    // new typeID increasing by 1, otherwise we use the typeID received from
    // the Master Catalog
    if (typeIDFromMasterCatalog == -1)
      typeCode = 8192 + allTypeNames.size();
    else
      typeCode = typeIDFromMasterCatalog;

    PDB_COUT << "Id Assigned to type " << typeName << " was "
             << std::to_string(typeCode) << endl;

    allTypeNames[typeName] = typeCode;
    allTypeCodes[typeCode] = typeName;

    vector<string> typeNames;
    vector<int> typeCodes;

    for (auto &pair : allTypeNames) {
      typeNames.push_back(pair.first);
      typeCodes.push_back(pair.second);
    }

    const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};
    Handle<CatalogUserTypeMetadata> objectMetadata =
        makeObject<CatalogUserTypeMetadata>();

    PDB_COUT << "before calling registerUserDefinedObject with typeCode="
             << typeCode << endl;

    // register the bytes of the shared library along its metadata in the
    // catalog
    pdbCatalog->registerUserDefinedObject(
        typeCode, objectMetadata, std::string(soFile.begin(), soFile.end()),
        typeName, catalogDirectory + "/tmp_so_files/" + typeName + ".so",
        "data_types", errMsg);

    return typeCode;
  } else
    return allTypeNames[typeName];
}

// deletes metadata about a Set in the Catalog
bool CatalogServer::deleteSet(std::string databaseName, std::string setName,
                              std::string &errMsg) {

  // prepares the key
  string setUniqueId = databaseName + "." + setName;
  PDB_COUT << "Deleting set " << setUniqueId << endl;

  // make sure the Database for this set is registered
  if (isDatabaseRegistered(databaseName) == false) {
    errMsg = "Database does not exist.\n";
    return false;
  }

  // make sure the set is registered
  if (isSetRegistered(databaseName, setName) == false) {
    errMsg = "Set doesn't exist " + databaseName + "." + setName;
    return false;
  }

  Handle<CatalogSetMetadata> metadataObject = makeObject<CatalogSetMetadata>();

  string _setName(databaseName + "." + setName);
  String setKeyCatalog = String(_setName);
  String setNameCatalog = String(setName);
  String dbName(databaseName);

  int catalogType = PDBCatalogMsgType::CatalogPDBSet;
  String setId = String(pdbCatalog->itemName2ItemId(catalogType, _setName));

  // populates object metadata
  metadataObject->setItemId(setId);
  metadataObject->setItemKey(setKeyCatalog);
  metadataObject->setItemName(setNameCatalog);
  metadataObject->setDBName(dbName);

  // deletes Set metadata in the catalog
  pdbCatalog->deleteMetadataInCatalog(metadataObject, catalogType, errMsg);

  // prepares Database object to update the set deletion
  Handle<CatalogDatabaseMetadata> dbMetadataObject =
      makeObject<CatalogDatabaseMetadata>();

  Handle<Vector<CatalogDatabaseMetadata>> databaseItems =
      makeObject<Vector<CatalogDatabaseMetadata>>();

  catalogType = PDBCatalogMsgType::CatalogPDBDatabase;

  // retrieves the metadata for this Set's Database
  if (pdbCatalog->getMetadataFromCatalog(false, databaseName, databaseItems,
                                         errMsg, catalogType) == false)
    PDB_COUT << errMsg << endl;

  for (int i = 0; i < (*databaseItems).size(); i++) {
    if ((*databaseItems)[i].getItemKey().c_str() == databaseName)
      *dbMetadataObject = (*databaseItems)[i];
  }

  // deletes the Set
  (*dbMetadataObject).deleteSet(setNameCatalog);

  // updates the corresponding database metadata
  if (!pdbCatalog->updateMetadataInCatalog(dbMetadataObject, catalogType,
                                           errMsg)) {
    return false;
  }

  // after it updated the database metadata in the local catalog, if this is the
  // master catalog iterate over all nodes in the cluster and broadcast the
  // update to the distributed copies of the catalog
  if (isMasterCatalogServer) {
    // map to capture the results of broadcasting the Set deletion
    map<string, pair<bool, string>> updateResults;

    Handle<CatDeleteSetRequest> setToRemove =
        makeObject<CatDeleteSetRequest>(databaseName, setName);

    // first, broadcasts the metadata of the removed set to all copies of the
    // catalog in the cluster, removing this Set
    broadcastCatalogDelete(setToRemove, updateResults, errMsg);

    for (auto &item : updateResults) {
      PDB_COUT << "Set Metadata in node IP: " << item.first
               << ((item.second.first == true)
                       ? " updated correctly!"
                       : " couldn't be updated due to error: ")
               << item.second.second << endl;
    }

    // map to capture the results of broadcasting the DB update
    map<string, pair<bool, string>> updateSetResults;

    // second, broadcasts the metadata of the DB to which this set has been
    // removed from, updating all distributed copies of the catalog
    broadcastCatalogUpdate(dbMetadataObject, updateSetResults, errMsg);

    for (auto &item : updateSetResults) {
      PDB_COUT << "DB Metadata in node IP: " << item.first
               << ((item.second.first == true)
                       ? "updated correctly!"
                       : "couldn't be updated due to error: ")
               << item.second.second << endl;
    }
    PDB_COUT << "******************* deleteSet step completed!!!!!!!" << endl;
  } else {
    PDB_COUT << "This is not Master Catalog Node, thus metadata was only "
                "registered locally!"
             << endl;
  }

  return true;
}

// Adds Metadata about a Set that has been created into the Catalog
bool CatalogServer::addSet(int16_t typeIdentifier, std::string databaseName,
                           std::string setName, std::string &errMsg) {
  auto begin = std::chrono::high_resolution_clock::now();

  // make sure we are only adding to an existing database
  if (isDatabaseRegistered(databaseName) == false) {
    errMsg = "Database does not exist.\n";
    return false;
  }

  // make sure that the set does not exist
  string setUniqueId = databaseName + "." + setName;
  if (isSetRegistered(databaseName, setName) == true) {
    errMsg = "Set already exists.\n";
    return false;
  }

  // make sure that type associated to this Set exists
  if (typeIdentifier >= 8192 && allTypeCodes.count(typeIdentifier) == 0) {
    errMsg = "Type code does not exist.\n";
    PDB_COUT << errMsg << "TypeId=" << std::to_string(typeIdentifier) << endl;
    return false;
  }

  PDB_COUT << "...... Calling CatalogServer :: addSet" << endl;

  // add the Type Id to the map of databases and sets
  setTypes[make_pair(databaseName, setName)] = typeIdentifier;

  // searches for the Type Name given its Type ID
  string typeNameStr = searchForObjectTypeName(typeIdentifier);
  PDB_COUT << "Got typeName=" << typeNameStr << endl;

  if (typeNameStr == "") {
    errMsg = "TypeName doesn not exist";
    return false;
  }

  String typeName(typeNameStr);

  auto afterChecks = std::chrono::high_resolution_clock::now();

  PDB_COUT << "TypeID for Set with dbName=" << databaseName
           << " and setName=" << setName << " is "
           << std::to_string(typeIdentifier) << endl;

  Handle<CatalogSetMetadata> metadataObject = makeObject<CatalogSetMetadata>();

  // gets the Database Id
  int catalogType = PDBCatalogMsgType::CatalogPDBDatabase;
  string dbId = pdbCatalog->itemName2ItemId(catalogType, databaseName);


  catalogType = PDBCatalogMsgType::CatalogPDBRegisteredObject;

  string typeId = std::to_string(typeIdentifier);

  // creates Strings
  String setKeyCatalog = String(setUniqueId);
  String setNameCatalog = String(setName);
  String dbName(databaseName);

  // populates object metadata
  metadataObject->setItemKey(setKeyCatalog);
  metadataObject->setItemName(setNameCatalog);

  // retrieves the Set Id
  catalogType = PDBCatalogMsgType::CatalogPDBDatabase;
  String _dbId = String(pdbCatalog->itemName2ItemId(catalogType, databaseName));

  metadataObject->setDBId(_dbId);
  metadataObject->setDBName(dbName);

  String _typeId = String(std::to_string(typeIdentifier));

  metadataObject->setTypeId(_typeId);
  metadataObject->setTypeName(typeName);

  auto beforeCallAddUpdate = std::chrono::high_resolution_clock::now();

  catalogType = PDBCatalogMsgType::CatalogPDBSet;

  // adds metadata to the catalog if this is a new item,
  // otherwise updates existing metadata
  if (isSetRegistered(dbName, setName) == false) {
    pdbCatalog->addMetadataToCatalog(metadataObject, catalogType,
                                     errMsg);
  } else {
    pdbCatalog->updateMetadataInCatalog(metadataObject, catalogType, errMsg);
  }

  auto afterCallAddUpdate = std::chrono::high_resolution_clock::now();

  // prepares object for the Database metadata
  catalogType = PDBCatalogMsgType::CatalogPDBDatabase;
  Handle<CatalogDatabaseMetadata> dbMetadataObject =
      makeObject<CatalogDatabaseMetadata>();

  // retrieves the metadata for this Set's Database
  Handle<Vector<CatalogDatabaseMetadata>> databaseItems =
      makeObject<Vector<CatalogDatabaseMetadata>>();

  if (pdbCatalog->getMetadataFromCatalog(false, databaseName, databaseItems,
                                         errMsg, catalogType) == false)
    PDB_COUT << errMsg << endl;

  for (int i = 0; i < (*databaseItems).size(); i++) {
    if ((*databaseItems)[i].getItemKey().c_str() == databaseName)
      *dbMetadataObject = (*databaseItems)[i];
  }

  auto aftergetMetaFromCat = std::chrono::high_resolution_clock::now();

  (*dbMetadataObject).addSet(setNameCatalog);
  (*dbMetadataObject).addType(typeName);

  // updates the corresponding database metadata
  if (isDatabaseRegistered(databaseName) == true) {
    catalogType = PDBCatalogMsgType::CatalogPDBDatabase;
    if (pdbCatalog->updateMetadataInCatalog(dbMetadataObject, catalogType,
                                            errMsg) == true)
      PDB_COUT << "DB Update Set metadata OK" << endl;
    else {
      PDB_COUT << "DB Update metadata Set Error: " << errMsg << endl;
      return false;
    }
  } else {
    return false;
  }

  auto afterUpdateMeta = std::chrono::high_resolution_clock::now();
  auto broadCast1 = afterUpdateMeta;
  auto broadCast2 = afterUpdateMeta;

  // after it updated the database metadata in the local catalog, if this is the
  // master catalog iterate over all nodes in the cluster and broadcast the
  // update to the distributed copies of the catalog
  if (isMasterCatalogServer) {
    // map to capture the results of broadcasting the Set insertion
    map<string, pair<bool, string>> updateResults;

    // first, broadcasts the metadata of the new set to the distributed copies
    // of the catalog updating metaddata of the new Set
    broadcastCatalogUpdate(metadataObject, updateResults, errMsg);
    broadCast1 = std::chrono::high_resolution_clock::now();

    for (auto &item : updateResults) {
      PDB_COUT << "Set Metadata broadcasted to node IP: "
               << item.first + ((item.second.first == true)
                                    ? " updated correctly!"
                                    : " couldn't be updated due to error: ")
               << item.second.second << endl;
    }

    // map to capture the results of broadcasting the DB update
    map<string, pair<bool, string>> updateSetResults;

    // second, broadcasts the metadata of the DB to which this set has been
    // added, updating the distributed copies of the catalog
    broadcastCatalogUpdate(dbMetadataObject, updateSetResults, errMsg);
    broadCast2 = std::chrono::high_resolution_clock::now();

    for (auto &item : updateSetResults) {
      PDB_COUT << "DB Metadata broadcasted to node IP: "
               << item.first + ((item.second.first == true)
                                    ? " updated correctly!"
                                    : " couldn't be updated due to error: ")
               << item.second.second << endl;
    }
    PDB_COUT << "******************* addSet step completed!!!!!!!" << endl;
  } else {
    PDB_COUT << "This is not Master Catalog Node, thus metadata was only "
                "registered locally!"
             << endl;
  }

  auto end = std::chrono::high_resolution_clock::now();

  PDB_COUT << "Time Duration for check registration:\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      afterChecks - begin)
                      .count())
           << " secs." << endl;
  PDB_COUT << "Time Duration for Setting Metadata values:\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      beforeCallAddUpdate - afterChecks)
                      .count())
           << " secs." << endl;
  PDB_COUT << "Time Duration for addMetadataToCatalog SET:\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      afterCallAddUpdate - beforeCallAddUpdate)
                      .count())
           << " secs." << endl;
  PDB_COUT << "Time Duration for getMetadataFromCatalog call:\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      aftergetMetaFromCat - afterCallAddUpdate)
                      .count())
           << " secs." << endl;
  PDB_COUT << "Time Duration for Updte DB Metadata:\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      afterUpdateMeta - aftergetMetaFromCat)
                      .count())
           << " secs." << endl;
  PDB_COUT << "Time Duration for broadcastCatalogUpdate SET:\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      broadCast1 - afterUpdateMeta)
                      .count())
           << " secs." << endl;
  PDB_COUT << "Time Duration for broadcastCatalogUpdate DB:\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      broadCast2 - broadCast1)
                      .count())
           << " secs." << endl;
  PDB_COUT << "------>Time Duration to AddSet\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      end - begin)
                      .count())
           << " secs." << endl;

  return true;
}

// add Metadata about a new Database into the Catalog
bool CatalogServer::addDatabase(std::string databaseName, std::string &errMsg) {

  // don't add a database that is already registered
  if (isDatabaseRegistered(databaseName) == true) {
    errMsg = "Database name already exists.\n";
    return false;
  }

  PDB_COUT << "...... Calling CatalogServer :: addDatabase" << endl;

  int catalogType = PDBCatalogMsgType::CatalogPDBDatabase;
  Handle<CatalogDatabaseMetadata> metadataObject =
      makeObject<CatalogDatabaseMetadata>();


  // populates object metadata
  String dbName = String(databaseName);
  metadataObject->setItemName(dbName);

  // adds Metadata into the Catalog
  if (isDatabaseRegistered(databaseName) == false) {
    pdbCatalog->addMetadataToCatalog(metadataObject, catalogType,
                                     errMsg);
  } else {
    pdbCatalog->updateMetadataInCatalog(metadataObject, catalogType, errMsg);
  }

  // after it registered the database metadata in the local catalog, if this is
  // the master catalog iterate over all nodes in the cluster and broadcast the
  // update to the distributed copies of the catalog
  if (isMasterCatalogServer) {

    // get the results of each broadcast
    map<string, pair<bool, string>> updateResults;

    broadcastCatalogUpdate(metadataObject, updateResults, errMsg);

    for (auto &item : updateResults) {
      PDB_COUT << "DB metadata broadcasted to node IP: "
               << item.first + ((item.second.first == true)
                                    ? " updated correctly!"
                                    : " couldn't be updated due to error: ")
               << item.second.second << endl;
    }
    PDB_COUT << "******************* addDatabase step completed!!!!!!!" << endl;
  } else {
    PDB_COUT << "This is not Master Catalog Node, thus metadata was only "
                "registered locally!"
             << endl;
  }

  return true;
}

// delete Metadata about a Database into the Catalog
bool CatalogServer::deleteDatabase(std::string databaseName,
                                   std::string &errMsg) {

  // don't delete a database that doesn't exist
  if (isDatabaseRegistered(databaseName) == false) {
    errMsg = "Database does not exist.\n";
    return false;
  }

  int catalogType = PDBCatalogMsgType::CatalogPDBDatabase;
  Handle<CatalogDatabaseMetadata> dbMetadataObject =
      makeObject<CatalogDatabaseMetadata>();
  Handle<Vector<CatalogDatabaseMetadata>> vectorResultItems =
      makeObject<Vector<CatalogDatabaseMetadata>>();

  if (pdbCatalog->getMetadataFromCatalog(false, databaseName, vectorResultItems,
                                         errMsg, catalogType) == false) {
    errMsg = "Database does not exist.\n";
    return false;
  }

  if ((*vectorResultItems).size() == 0) {
    errMsg = "Database does not exist.\n";
    return false;
  }

  *dbMetadataObject = (*vectorResultItems)[0];

  // Delete the sets of this Database from the Catalog
  for (int i = 0; i < dbMetadataObject->getListOfSets()->size(); i++) {
    if (!deleteSet(databaseName, (*dbMetadataObject->getListOfSets())[i],
                   errMsg)) {
      errMsg = "Failed to delete set ";
      errMsg += (*dbMetadataObject->getListOfSets())[i].c_str();
    }
  }

  // after it deleted the database metadata in the local catalog, if this is the
  // master catalog iterate over all nodes in the cluster and broadcast the
  // update to the distributed copies of the catalog
  if (isMasterCatalogServer) {
    // get the results of each broadcast
    map<string, pair<bool, string>> updateResults;
    errMsg = "";

    Handle<CatDeleteDatabaseRequest> databaseToRemove =
        makeObject<CatDeleteDatabaseRequest>(databaseName);
    broadcastCatalogDelete(databaseToRemove, updateResults, errMsg);

    for (auto &item : updateResults) {
      // adds node info to database metadata
      PDB_COUT << "DB metadata broadcasted to node IP: " << item.first
               << ((item.second.first == true)
                       ? " deleted correctly!"
                       : " couldn't be deleted due to error: ")
               << item.second.second << endl;
    }
  } else {
    PDB_COUT << "This is not Master Catalog Node, thus metadata was only "
                "registered locally!"
             << endl;
  }

  // deletes the metadata in the Catalog
  return pdbCatalog->deleteMetadataInCatalog(dbMetadataObject, catalogType,
                                             errMsg);
}

// destructor
CatalogServer::~CatalogServer() { pthread_mutex_destroy(&workingMutex); }

// constructor
CatalogServer::CatalogServer(std::string catalogDirectoryIn,
                             bool isMasterCatalogServer, string masterIPValue,
                             int masterPortValue) {
  PDBLoggerPtr catalogLogger = make_shared<PDBLogger>("catalogLogger");
  catServerLogger = make_shared<pdb::PDBLogger>("catalogServer.log");

  auto begin = std::chrono::high_resolution_clock::now();

  masterIP = masterIPValue;
  masterPort = masterPortValue;

  catalogClientConnectionToMasterCatalogServer =
      CatalogClient(masterPort, masterIP,
                    make_shared<pdb::PDBLogger>("clientCatalogToServerLog"));

  const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};
  _allNodesInCluster = makeObject<Vector<CatalogNodeMetadata>>();
  _setTypes = makeObject<Vector<CatalogSetMetadata>>();
  _allDatabases = makeObject<Vector<CatalogDatabaseMetadata>>();
  _udfsValues = makeObject<Vector<CatalogUserTypeMetadata>>();

  this->isMasterCatalogServer = isMasterCatalogServer;

  catalogDirectory = catalogDirectoryIn;
  PDB_COUT << "Catalog Server ctor is Master Catalog= "
           << std::to_string(this->isMasterCatalogServer) << endl;

  // creates instance of catalog
  pdbCatalog = make_shared<PDBCatalog>(catalogLogger, catalogDirectory);

  // retrieves catalog metadata from an SQLite storage and loads metadata into
  // memory
  pdbCatalog->open();

  pthread_mutex_init(&workingMutex, nullptr);

  PDB_COUT << "Loading catalog metadata." << endl;

  string errMsg;
  string emptyString("");

  // retrieves metadata for user-defined types from SQLite storage and loads
  // them into memory
  if (pdbCatalog->getMetadataFromCatalog(
          false, emptyString, _udfsValues, errMsg,
          PDBCatalogMsgType::CatalogPDBRegisteredObject) == false)

    PDB_COUT << errMsg << endl;

  for (int i = 0; i < (*_udfsValues).size(); i++) {
    string _typeName = (*_udfsValues)[i].getItemKey().c_str();
    int16_t _typeId = (int16_t)atoi((*_udfsValues)[i].getObjectID().c_str());

    allTypeNames[_typeName] = _typeId;
    allTypeCodes[_typeId] = _typeName;
  }

  // retrieves metadata for databases from SQLite storage and loads them into
  // memory
  if (pdbCatalog->getMetadataFromCatalog(
          false, emptyString, _allDatabases, errMsg,
          PDBCatalogMsgType::CatalogPDBDatabase) == false)

    PDB_COUT << errMsg << endl;

  for (int i = 0; i < (*_allDatabases).size(); i++) {
    string _dbName = (*_allDatabases)[i].getItemKey().c_str();
    for (int j = 0; j < (*(*_allDatabases)[i].getListOfSets()).size(); j++) {
      string _setName = (*(*_allDatabases)[i].getListOfSets())[j].c_str();
      string _typeName = (*(*_allDatabases)[i].getListOfTypes())[j].c_str();

      PDB_COUT << "Database " << _dbName << " has set " << _setName
               << " and type " << _typeName << endl;

      // populates information about types and sets for a given database
      PDB_COUT << "Adding type= " << _typeName << " db= " << _dbName
               << " _set=" << _setName << " typeId= "
               << string(pdbCatalog->getUserDefinedTypesList()[_typeName]
                             .getObjectID())
               << endl;

      setTypes[make_pair(_dbName, _setName)] =
          (int16_t)std::atoi(pdbCatalog->getUserDefinedTypesList()[_typeName]
                                 .getObjectID()
                                 .c_str());
    }
  }

  // retrieves metadata for nodes in the cluster from SQLite storage and loads
  // them into memory
  if (pdbCatalog->getMetadataFromCatalog(
          false, emptyString, _allNodesInCluster, errMsg,
          PDBCatalogMsgType::CatalogPDBNode) == false)

    PDB_COUT << errMsg << endl;

  for (int i = 0; i < (*_allNodesInCluster).size(); i++) {
    string _nodeAddress = (*_allNodesInCluster)[i].getItemId().c_str();
    string _nodeIP = (*_allNodesInCluster)[i].getNodeIP().c_str();
    int _nodePort = (*_allNodesInCluster)[i].getNodePort();
    string _nodeName = (*_allNodesInCluster)[i].getItemName().c_str();
    string _nodeType = (*_allNodesInCluster)[i].getNodeType().c_str();
    int status = (*_allNodesInCluster)[i].getNodeStatus();

    PDB_COUT << _nodeAddress << " | " << _nodeIP << " | "
             << std::to_string(_nodePort) << " | " << _nodeName << " | "
             << _nodeType << " | " << std::to_string(status) << endl;

    allNodesInCluster.push_back(_nodeAddress);
  }

  auto end = std::chrono::high_resolution_clock::now();

  PDB_COUT << "Catalog Metadata successfully loaded!" << endl;
  PDB_COUT << "--------->Populate CatalogServer Metadata : "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      end - begin)
                      .count())
           << " secs." << endl;
}

// invokes a method to retrieve metadata that has changed since a given
// timestamp
void CatalogServer::printCatalog(
    Handle<CatalogPrintMetadata> &metadataToPrint) {

  string itemKey = metadataToPrint->getItemName().c_str();
  string categoryToPrint = metadataToPrint->getCategoryToPrint().c_str();
  string timeStamp = metadataToPrint->getTimeStamp().c_str();

  string resultToPrint;
  string errorMsg;

  if (categoryToPrint.compare("all") == 0) {
      pdbCatalog->listNodesInCluster(resultToPrint, errorMsg);
      pdbCatalog->listRegisteredDatabases(resultToPrint, errorMsg);
      pdbCatalog->listUserDefinedTypes(resultToPrint, errorMsg);
  } else {

      if (categoryToPrint.compare("databases") == 0)
        pdbCatalog->listRegisteredDatabases(resultToPrint, errorMsg);

      if (categoryToPrint.compare("sets") == 0)
        pdbCatalog->listRegisteredSetsForADatabase(resultToPrint, itemKey,
                                                   errorMsg);

      if (categoryToPrint.compare("nodes") == 0)
        pdbCatalog->listNodesInCluster(resultToPrint, errorMsg);

      if (categoryToPrint.compare("udts") == 0)
        pdbCatalog->listUserDefinedTypes(resultToPrint, errorMsg);
  }

  metadataToPrint->setMetadataToPrint(resultToPrint);

}

// add metadata about a Node in the cluster
bool CatalogServer::addNodeMetadata(Handle<CatalogNodeMetadata> &nodeMetadata,
                                    std::string &errMsg) {

  string _nodeIP = nodeMetadata->getNodeIP().c_str();
  string nodeAddress = _nodeIP + ":" + to_string(nodeMetadata->getNodePort());

  // don't add a node that is already registered
  if (isNodeRegistered(nodeAddress) == true) {
    errMsg = "NodeAddress " + nodeAddress + " is already registered.\n";
    return false;
  }

  // add the node info to container
  allNodesInCluster.push_back(nodeAddress);

  int metadataCategory = PDBCatalogMsgType::CatalogPDBNode;
  Handle<CatalogNodeMetadata> metadataObject =
      makeObject<CatalogNodeMetadata>();

  *metadataObject = *nodeMetadata;

  pdbCatalog->addMetadataToCatalog(metadataObject,
                                   metadataCategory, errMsg);

  PDB_COUT << "Node metadata was properly registered in the Catalog!" << endl;
  
  return true;
}

// adds Metadata of a new Database to the Catalog
bool CatalogServer::addDatabaseMetadata(
    Handle<CatalogDatabaseMetadata> &dbMetadata, std::string &errMsg) {
  string dbName = dbMetadata->getItemName().c_str();

  // don't add a database that is already registered
  if (isDatabaseRegistered(dbName) == true) {
    errMsg = "Db name: " + dbName + " is already registered.\n";
    return false;
  }

  int metadataCategory = PDBCatalogMsgType::CatalogPDBDatabase;
  Handle<CatalogDatabaseMetadata> metadataObject =
      makeObject<CatalogDatabaseMetadata>();

  *metadataObject = *dbMetadata;

  pdbCatalog->addMetadataToCatalog(metadataObject,
                                   metadataCategory, errMsg);

  // after it registered the Database metadata in the local catalog, if this is
  // the master catalog iterate over all nodes in the cluster and broadcast the
  // insert to the distributed copies of the catalog
  if (isMasterCatalogServer) {
    // get the results of each broadcast
    map<string, pair<bool, string>> updateResults;
    errMsg = "";
    broadcastCatalogUpdate(metadataObject, updateResults, errMsg);

    for (auto &item : updateResults) {
      PDB_COUT << "Node IP: " << item.first
               << ((item.second.first == true)
                       ? " updated correctly!"
                       : " couldn't be updated due to error: ")
               << item.second.second << endl;
    }
  } else {
    PDB_COUT << "This is not Master Catalog Node, thus metadata was only "
                "registered locally!"
             << endl;
  }

  return true;
}

// updates metadata about a Database that has changed in the catalog
bool CatalogServer::updateDatabaseMetadata(
    Handle<CatalogDatabaseMetadata> &dbMetadata, std::string &errMsg) {

  string dbName = dbMetadata->getItemName().c_str();

  int metadataCategory = PDBCatalogMsgType::CatalogPDBDatabase;
  Handle<CatalogDatabaseMetadata> metadataObject =
      makeObject<CatalogDatabaseMetadata>();
  *metadataObject = *dbMetadata;

  pdbCatalog->updateMetadataInCatalog(metadataObject, metadataCategory, errMsg);

  // after it updates the Database metadata in the local catalog, if this is the
  // master catalog iterate over all nodes in the cluster and broadcast the
  // insert to the distributed copies of the catalog
  if (isMasterCatalogServer) {
    // get the results of each broadcast
    map<string, pair<bool, string>> updateResults;
    errMsg = "";
    broadcastCatalogUpdate(metadataObject, updateResults, errMsg);

    for (auto &item : updateResults) {
      PDB_COUT << "Node IP: "
               << item.first + ((item.second.first == true)
                                    ? " updated correctly!"
                                    : " couldn't be updated due to error: ")
               << item.second.second << endl;
    }
  } else {
    PDB_COUT << "This is not Master Catalog Node, thus metadata was only "
                "registered locally!"
             << endl;
  }

  return true;
}

// adds Metadata of a new Set into the Catalog
bool CatalogServer::addSetMetadata(Handle<CatalogSetMetadata> &setMetadata,
                                   std::string &errMsg) {
  auto begin = std::chrono::high_resolution_clock::now();

  // gets the set name
  string setName = string(setMetadata->getItemName().c_str());
  // gets the database name
  string dbName = string(setMetadata->getDBName().c_str());
  // gets the type Id
  int16_t typeId = (int16_t)atoi((*setMetadata).getObjectTypeId().c_str());

  // don't add a set that is already registered
  if (isSetRegistered(dbName, setName) == true) {
    errMsg =
        "Set name: " + dbName + "." + setName + " is already registered.\n";
    return false;
  }

  PDB_COUT << "inserting set-----------------> dbName= " << dbName
           << " setName " << setName << " id " << std::to_string(typeId)
           << endl;
  setTypes.insert(make_pair(make_pair(dbName, setName), typeId));

  int metadataCategory = PDBCatalogMsgType::CatalogPDBSet;
  Handle<CatalogSetMetadata> metadataObject = makeObject<CatalogSetMetadata>();

  *metadataObject = *setMetadata;

  auto beforeaddMetaToCat = std::chrono::high_resolution_clock::now();

  PDB_COUT << "Adding set metadata for set " << setName << endl;
  pdbCatalog->addMetadataToCatalog(metadataObject, 
                                   metadataCategory, errMsg);

  auto afteraddMetToCat = std::chrono::high_resolution_clock::now();
  auto beforeBroadCatUpdate = afteraddMetToCat;
  auto afterBroadCatUpdate = afteraddMetToCat;

  // after it registered the Set metadata in the local catalog, if this is the
  // master catalog iterate over all nodes in the cluster and broadcast the
  // insert to the distributed copies of the catalog
  if (isMasterCatalogServer) {
    // get the results of each broadcast
    map<string, pair<bool, string>> updateResults;

    beforeBroadCatUpdate = std::chrono::high_resolution_clock::now();

    broadcastCatalogUpdate(metadataObject, updateResults, errMsg);

    afterBroadCatUpdate = std::chrono::high_resolution_clock::now();

    for (auto &item : updateResults) {
      PDB_COUT << "Node IP: "
               << item.first + ((item.second.first == true)
                                    ? " updated correctly!"
                                    : " couldn't be updated due to error: ")
               << item.second.second << endl;
    }
  } else {
    PDB_COUT << "This is not Master Catalog Node, thus metadata was only "
                "registered locally!"
             << endl;
  }
  auto beforeReturn = std::chrono::high_resolution_clock::now();

  PDB_COUT << "Time Duration for check and copy:\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      beforeaddMetaToCat - begin)
                      .count())
           << " secs." << endl;
  PDB_COUT << "Time Duration for adding metadata to catalog:\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      afteraddMetToCat - beforeaddMetaToCat)
                      .count())
           << " secs." << endl;
  PDB_COUT << "Time Duration for check isMasterCatalog:\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      beforeBroadCatUpdate - afteraddMetToCat)
                      .count())
           << " secs." << endl;
  PDB_COUT << "Time Duration for broadcast update:\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      afterBroadCatUpdate - beforeBroadCatUpdate)
                      .count())
           << " secs." << endl;
  PDB_COUT << "Time Duration before return:\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      beforeReturn - afterBroadCatUpdate)
                      .count())
           << " secs." << endl;
  PDB_COUT << "------>Time Duration to Complete addSetMetadata\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      beforeReturn - begin)
                      .count())
           << " secs." << endl;

  return true;
}

// Adds the IP of a node to a given set
bool CatalogServer::addNodeToSet(std::string nodeIP, std::string databaseName,
                                 std::string setName, std::string &errMsg) {
  auto begin = std::chrono::high_resolution_clock::now();

  // make sure we are only adding to an existing database
  if (isDatabaseRegistered(databaseName) == false) {
    errMsg = "Database does not exist.\n";
    return false;
  }

  // make sure we are only adding to an existing Set
  if (isSetRegistered(databaseName, setName) == false) {
    errMsg = "Set does not exists.\n";
    return false;
  }

  auto afterRegisteredCheck = std::chrono::high_resolution_clock::now();

  PDB_COUT << "...... Calling CatalogServer :: addNodeToSet" << endl;

  // prepares object for the DB metadata
  int catalogType = PDBCatalogMsgType::CatalogPDBDatabase;
  Handle<CatalogDatabaseMetadata> dbMetadataObject =
      makeObject<CatalogDatabaseMetadata>();

  Handle<Vector<CatalogDatabaseMetadata>> databaseItems =
      makeObject<Vector<CatalogDatabaseMetadata>>();

  if (pdbCatalog->getMetadataFromCatalog(false, databaseName, databaseItems,
                                         errMsg, catalogType) == false)
    PDB_COUT << errMsg << endl;

  for (int i = 0; i < (*databaseItems).size(); i++) {
    if ((*databaseItems)[i].getItemKey().c_str() == databaseName)
      *dbMetadataObject = (*databaseItems)[i];
  }

  String setNameCatalog(setName);
  String nodeID(nodeIP);

  // add Set to Map
  (*dbMetadataObject).addSetToMap(setNameCatalog, nodeID);

  // updates the corresponding database metadata
  catalogType = PDBCatalogMsgType::CatalogPDBDatabase;

  auto afteraddSetToMap = std::chrono::high_resolution_clock::now();

  // if database exists update its metadata
  if (isDatabaseRegistered(databaseName) == true) {
    PDB_COUT << ".......... Invoking updateMetadataInCatalog key: "
             << databaseName << endl;
    if (pdbCatalog->updateMetadataInCatalog(dbMetadataObject, catalogType,
                                            errMsg) == true)
      PDB_COUT << "DB Update Set metadata OK" << endl;
    else {
      PDB_COUT << "DB Update metadata Set Error: " << errMsg << endl;
      return false;
    }
  } else {
    return false;
  }

  auto afterUpdateDB = std::chrono::high_resolution_clock::now();

  // after it registered the Set metadata in the local catalog, if this is the
  // master catalog iterate over all nodes in the cluster and broadcast the
  // insert to the distributed copies of the catalog
  if (isMasterCatalogServer) {
    PDB_COUT << "About to broadcast addition of node to set in the cluster: "
             << endl;

    // map to capture the results of broadcasting the DB update
    map<string, pair<bool, string>> updateSetResults;

    if (broadcastCatalogUpdate(dbMetadataObject, updateSetResults, errMsg)) {
      PDB_COUT << " Broadcasting DB updated Ok. " << endl;
    } else {
      PDB_COUT << " Error broadcasting DB update." << endl;
    }
    for (auto &item : updateSetResults) {
      PDB_COUT << "Node IP: "
               << item.first + ((item.second.first == true)
                                    ? " updated correctly!"
                                    : " couldn't be updated due to error: ")
               << item.second.second << endl;
    }
  } else {
    PDB_COUT << "This is not Master Catalog Node, thus metadata was only "
                "registered locally!"
             << endl;
  }

  auto afterBroadcast = std::chrono::high_resolution_clock::now();
  auto addNodeToSet = std::chrono::high_resolution_clock::now();

  PDB_COUT << "Time Duration for check registration:\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      afterRegisteredCheck - begin)
                      .count())
           << " secs." << endl;
  PDB_COUT << "Time Duration for adding set to map:\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      afteraddSetToMap - afterRegisteredCheck)
                      .count())
           << " secs." << endl;
  PDB_COUT << "Time Duration for updating db Metadata:\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      afterUpdateDB - afteraddSetToMap)
                      .count())
           << " secs." << endl;
  PDB_COUT << "Time Duration for broadcast update:\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      afterBroadcast - afterUpdateDB)
                      .count())
           << " secs." << endl;
  PDB_COUT << "------>Time Duration to Complete addNodeToSet\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      addNodeToSet - begin)
                      .count())
           << " secs." << endl;

  return true;
}

// adds Node Information to a registered Database
bool CatalogServer::addNodeToDB(std::string nodeIP, std::string databaseName,
                                std::string &errMsg) {
  // make sure we are only adding to an existing database
  if (isDatabaseRegistered(databaseName) == false) {
    errMsg = "Database does not exist.\n";
    return false;
  }

  PDB_COUT << "...... Calling CatalogServer :: addNodeToSet" << endl;

  // prepares data for the DB metadata
  int catalogType = PDBCatalogMsgType::CatalogPDBDatabase;
  Handle<CatalogDatabaseMetadata> dbMetadataObject =
      makeObject<CatalogDatabaseMetadata>();

  Handle<Vector<CatalogDatabaseMetadata>> databaseItems =
      makeObject<Vector<CatalogDatabaseMetadata>>();

  if (pdbCatalog->getMetadataFromCatalog(false, databaseName, databaseItems,
                                         errMsg, catalogType) == false)
    PDB_COUT << errMsg << endl;

  for (int i = 0; i < (*databaseItems).size(); i++) {
    if ((*databaseItems)[i].getItemKey().c_str() == databaseName)
      *dbMetadataObject = (*databaseItems)[i];
  }

  String nodeID(nodeIP);
  String dbName(databaseName);
  (*dbMetadataObject).addNodeToMap(nodeID, dbName);

  // updates the corresponding database metadata
  catalogType = PDBCatalogMsgType::CatalogPDBDatabase;

  // if database exists update its metadata
  if (isDatabaseRegistered(databaseName) == true) {
    PDB_COUT << ".......... Invoking updateMetadataInCatalog key: " +
                    databaseName
             << endl;
    if (pdbCatalog->updateMetadataInCatalog(dbMetadataObject, catalogType,
                                            errMsg) == true)
      PDB_COUT << "DB Update Set metadata OK" << endl;
    else {
      PDB_COUT << "DB Update metadata Set Error: " << errMsg << endl;
      return false;
    }

  } else {
    return false;
  }

  // after it registered the Set metadata in the local catalog, if this is the
  // mastervcatalog iterate over all nodes in the cluster and broadcast the
  // insert to the distributed copies of the catalog
  if (isMasterCatalogServer) {
    PDB_COUT << "About to broadcast addition of node to set in the cluster: "
             << endl;

    // map to capture the results of broadcasting the DB update
    map<string, pair<bool, string>> updateSetResults;

    // second, broadcasts the metadata of the DB to which this set has been
    // added, updating all distributed copies of the catalog in the cluster
    if (broadcastCatalogUpdate(dbMetadataObject, updateSetResults, errMsg)) {
      PDB_COUT << " Broadcasting DB updated Ok. " << endl;
    } else {
      PDB_COUT << " Error broadcasting DB update." << endl;
    }
    for (auto &item : updateSetResults) {
      PDB_COUT << "Node IP: "
               << item.first + ((item.second.first == true)
                                    ? " updated correctly!"
                                    : " couldn't be updated due to error: ")
               << item.second.second << endl;
    }

  } else {
    PDB_COUT << "This is not Master Catalog Node, thus metadata was only "
                "registered locally!"
             << endl;
  }

  return true;
}

// removes node information from a Set
bool CatalogServer::removeNodeFromSet(std::string nodeIP,
                                      std::string databaseName,
                                      std::string setName,
                                      std::string &errMsg) {
  string setUniqueId = databaseName + "." + setName;

  // make sure the database exists
  if (!isDatabaseRegistered(databaseName)) {
    errMsg = "Database does not exist.\n";
    PDB_COUT << errMsg << endl;
    return false;
  }

  // make sure the set exists
  if (!isSetRegistered(databaseName, setName)) {
    errMsg = "Set doesn't exist.\n";
    PDB_COUT << errMsg << endl;
    return false;
  }

  int catalogType = PDBCatalogMsgType::CatalogPDBDatabase;
  Handle<CatalogDatabaseMetadata> dbMetadataObject =
      makeObject<CatalogDatabaseMetadata>();

  Handle<Vector<CatalogDatabaseMetadata>> databaseItems =
      makeObject<Vector<CatalogDatabaseMetadata>>();

  if (!pdbCatalog->getMetadataFromCatalog(false, databaseName, databaseItems,
                                          errMsg, catalogType)) {
    PDB_COUT << errMsg << endl;
    return false;
  }

  if (databaseItems->size() != 1) {
    errMsg = "Could not find database " + databaseName;
    PDB_COUT << errMsg << endl;
    return false;
  }

  *dbMetadataObject = (*databaseItems)[0];

  dbMetadataObject->removeNodeFromSet(nodeIP, setName);

  // updates catalog
  if (pdbCatalog->updateMetadataInCatalog(dbMetadataObject, catalogType,
                                          errMsg)) {
    PDB_COUT << "DB Update Set metadata OK" << endl;
  } else {
    PDB_COUT << "DB Update metadata Set Error: " << errMsg << endl;
    return false;
  }

  // after it registered the Set metadata in the local catalog, if this is the
  // master catalog iterate over all nodes in the cluster and broadcast the
  // insert to the distributed copies of the catalog
  if (isMasterCatalogServer) {
    // map to capture the results of broadcasting the Set insertion
    map<string, pair<bool, string>> updateResults;

    // first, broadcasts the metadata of the new set to all local copies of the
    // catalog in the cluster, inserting the new item
    broadcastCatalogUpdate(dbMetadataObject, updateResults, errMsg);

    for (auto &item : updateResults) {
      PDB_COUT << "Node IP: " << item.first
               << ((item.second.first == true)
                       ? " updated correctly!"
                       : " couldn't be updated due to error: ")
               << item.second.second << endl;
    }

    // map to capture the results of broadcasting the DB update
    map<string, pair<bool, string>> updateSetResults;

    // second, broadcasts the metadata of the DB to which this set has been
    // added, updating all local copies of the catalog in the cluster
    broadcastCatalogUpdate(dbMetadataObject, updateSetResults, errMsg);

    for (auto &item : updateSetResults) {
      PDB_COUT << "DB Metadata broadcasted to Node IP: "
               << item.first + ((item.second.first == true)
                                    ? " updated correctly!"
                                    : " couldn't be updated due to error: ")
               << item.second.second << endl;
    }
  } else {
    PDB_COUT << "This is not Master Catalog Node, thus metadata was only "
                "registered locally!"
             << endl;
  }
  return true;
}

// removes a node ip from a registered Database
bool CatalogServer::removeNodeFromDB(std::string nodeIP,
                                     std::string databaseName,
                                     std::string &errMsg) {
  errMsg = "Remove node from db not implemented";
  return false;
}

// templated method for broadcasting a Catalog Update to nodes in the cluster
template <class Type>
bool CatalogServer::broadcastCatalogUpdate(
    Handle<Type> metadataToSend,
    map<string, pair<bool, string>> &broadcastResults, string &errMsg) {

  PDBLoggerPtr catalogLogger = make_shared<PDBLogger>("distCatalogLogger");

  auto beforeLoop = std::chrono::high_resolution_clock::now();

  for (auto &item : pdbCatalog->getListOfNodesInCluster()) {
    string nodeAddress = string(item.second.getNodeIP().c_str()) + ":" +
                         to_string(item.second.getNodePort());
    string nodeIP = item.second.getNodeIP().c_str();
    int nodePort = item.second.getNodePort();
    bool res = false;

    CatalogClient clusterCatalogClient =
        CatalogClient(nodePort, nodeIP, catalogLogger);

    if (string(item.second.getNodeType().c_str()).compare("master") != 0) {

      auto beforeRegGeneric = std::chrono::high_resolution_clock::now();

      // sends the request to a node in the cluster
      res =
          clusterCatalogClient.registerGenericMetadata(metadataToSend, errMsg);

      auto afterRegGeneric = std::chrono::high_resolution_clock::now();

      PDB_COUT << "Time Duration for registerGenericMetadata call to node "
               << nodeAddress + " \t"
               << std::to_string(
                      std::chrono::duration_cast<std::chrono::duration<float>>(
                          afterRegGeneric - beforeRegGeneric)
                          .count())
               << " secs." << endl;

      // adds the result of the update
      broadcastResults.insert(make_pair(nodeIP, make_pair(res, errMsg)));
    }
  }

  auto afterLoop = std::chrono::high_resolution_clock::now();

  PDB_COUT << "------>Time Duration to complete broadcastCatalogUpdate\t"
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      afterLoop - beforeLoop)
                      .count())
           << " secs." << endl;

  return true;
}

} // namespace

// templated method for broadcasting a Catalog Deletion to nodes in the cluster
template <class Type>
bool CatalogServer::broadcastCatalogDelete(
    Handle<Type> metadataToSend,
    map<string, pair<bool, string>> &broadcastResults, string &errMsg) {

  PDBLoggerPtr catalogLogger = make_shared<PDBLogger>("distCatalogLogger");

  for (auto &item : pdbCatalog->getListOfNodesInCluster()) {

    string nodeAddress = string(item.second.getNodeIP().c_str()) + ":" +
                         to_string(item.second.getNodePort());
    string nodeIP = item.second.getNodeIP().c_str();
    int nodePort = item.second.getNodePort();
    bool res = false;

    CatalogClient clusterCatalogClient =
        CatalogClient(nodePort, nodeIP, catalogLogger);

    if (string(item.second.getNodeType().c_str()).compare("master") != 0) {

      // sends the request to a node in the cluster
      res = clusterCatalogClient.deleteGenericMetadata(metadataToSend, errMsg);

      // adds the result of the update
      broadcastResults.insert(make_pair(nodeIP, make_pair(res, errMsg)));

    } else {
      PDB_COUT << "Don't broadcast to " + nodeAddress +
                      " because it has the master catalog."
               << endl;
    }
  }

  return true;
}

// checks if a Database is registered
bool CatalogServer::isDatabaseRegistered(string dbName) {
  int catalogType = PDBCatalogMsgType::CatalogPDBDatabase;
  string result("");

  return pdbCatalog->keyIsFound(catalogType, dbName, result);
}

// checks if a Set is registered
bool CatalogServer::isSetRegistered(string dbName, string setName) {
  int catalogType = PDBCatalogMsgType::CatalogPDBSet;
  string result("");
  string setUniqueId = dbName + "." + setName;
  return pdbCatalog->keyIsFound(catalogType, setUniqueId, result);
}

// checks if a Node is registered
bool CatalogServer::isNodeRegistered(string nodeIP) {
  int catalogType = PDBCatalogMsgType::CatalogPDBNode;
  string result("");
  PDB_COUT << "searching to register node " << nodeIP << endl;

  return pdbCatalog->keyIsFound(catalogType, nodeIP, result);
}

// checks if this is the Master Catalog
bool CatalogServer::getIsMasterCatalogServer() {
  return isMasterCatalogServer;
}

// sets this as a Master (true) or Worker (false) Catalog
void CatalogServer::setIsMasterCatalogServer(bool isMasterCatalogServerIn) {
  isMasterCatalogServer = isMasterCatalogServerIn;
}

/* Explicit instantiation to broadcast Catalog Updates for a Node */
template bool CatalogServer::broadcastCatalogUpdate(
    Handle<CatalogNodeMetadata> metadataToSend,
    map<string, pair<bool, string>> &broadcastResults,
    string &errMsg);

/* Explicit instantiation to broadcast Catalog Updates for a Database */
template bool CatalogServer::broadcastCatalogUpdate(
    Handle<CatalogDatabaseMetadata> metadataToSend,
    map<string, pair<bool, string>> &broadcastResults,
    string &errMsg);

/* Explicit instantiation to broadcast Catalog Updates for a Set */
template bool CatalogServer::broadcastCatalogUpdate(
    Handle<CatalogSetMetadata> metadataToSend,
    map<string, pair<bool, string>> &broadcastResults,
    string &errMsg);
