
#ifndef CATALOG_TESTS_CC
#define CATALOG_TESTS_CC

#include "CatSharedLibraryByNameRequest.h"
#include "Configuration.h"
#include "InterfaceFunctions.h"
#include "PDBClient.h"
#include "PDBVector.h"
#include "SharedEmployee.h"
#include "StorageClient.h"

#include <boost/program_options.hpp>

namespace po = boost::program_options;

using namespace std;
using namespace pdb;

int main(int numArgs, const char *args[]) {

  string serverAddress;

  po::options_description desc("Options");
  desc.add_options()("help", "produce help messages")(
      "conf", po::value<std::vector<std::string>>(), "configure files")(
      "command", po::value<string>(),
      "\nregister-type      Registers in the catalog a user-defined type "
      "contained in an .so "
      "dynamic library\n"
      "retrieve-type      Retrieves from the catalog a user-defined type "
      "stored as an .so "
      "dynamic library\n"
      "register-node      Registers in the catalog a user-defined type "
      "contained in an .so "
      "dynamic library\n"
      "retrieve-node      Retrieves from the catalog information about the "
      "nodes in the cluster\n"
      "register-db            Registers in the catalog a database\n"
      "retrieve-db            Retrieves from the catalog information about "
      "registered databases\n"
      "get-serialized-catalog Retrieves a serialized version of the catalog "
      "(as an SQLite file)\n"
      "print-catalog Retrieves information from the catalog as a string\n"
      "register-set       Registers in the catalog a set associated to a "
      "database\n"
      "retrieve-set       Retrieves from the catalog information about the "
      "registered sets\n"
      "shut-down          Shuts down the server\n")(
      "port", po::value<int>(), "frontEnd server listening port, default is "
                                "8108. In the case of Client this is the "
                                "frontEnd server to connect to")(
      "serverAddress", po::value<string>(),
      "IP address of the front end server, default is localhost")(
      "nodeIP", po::value<int>(),
      "ID of the node")("serverName", po::value<int>(), "name of the node")(
      "node-ip", po::value<string>(),
      "node ip to be registered in PDB catalog")(
      "node-port", po::value<int>(),
      "node port to be registered in PDB catalog")(
      "node-name", po::value<string>(),
      "node name to be registered in PDB catalog")(
      "node-type", po::value<string>(),
      "node type to be registered in PDB catalog")(
      "db-name", po::value<string>(),
      "database name to be registered in PDB catalog")(
      "type-name", po::value<string>(),
      "type name to be registered in PDB catalog")("type-id", po::value<int>(),
                                                   "type id of data type")(
      "set-name", po::value<string>(),
      "set name to be registered in PDB catalog")(
      "so-library-file", po::value<string>(),
      "name of .so library file to be registered, e.g. "
      "'libraries/libSharedEmployee.so'")(
      "timestamp", po::value<string>(),
      "timestamp in unix format for retrieving newer metadata")(
      "library-type", po::value<string>(),
      "can take two values: 'data-types' (for data) or 'metrics' (for "
      "metrics)")("register-type", po::value<string>(),
                  "Registers in the catalog a user-defined type contained in "
                  "an .so dynamic library")(
      "retrieve-type", po::value<string>(),
      "Retrieves from the catalog a user-defined type stored as an .so dynamic "
      "library")("register-node", po::value<string>(),
                 "Registers in the catalog a user-defined type contained in an "
                 ".so dynamic library")(
      "retrieve-node", po::value<string>(),
      "Retrieves from the catalog information about the nodes in the cluster")(
      "register-db", po::value<string>(),
      "Registers in the catalog a database")(
      "retrieve-db", po::value<string>(),
      "Retrieves from the catalog information about registered databases")(
      "print-catalog", po::value<string>(), "    Prints metadata from catalog")(
      "register-set", po::value<string>(),
      "Registers in the catalog a set associated to a database")(
      "retrieve-set", po::value<string>(),
      "Retrieves from the catalog information about the registered sets");

  po::positional_options_description p;
  p.add("conf", -1);
  po::variables_map vm;
  po::store(
      po::command_line_parser(numArgs, args).options(desc).positional(p).run(),
      vm);
  // po::store(po::parse_command_line(numArgs, args, desc), vm);
  po::notify(vm);

  ConfigurationPtr conf = make_shared<Configuration>();

  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }

  if (vm.count("conf")) {
    std::vector<std::string> files = vm["conf"].as<std::vector<std::string>>();
    for (std::string file : files) {
      std::cout << "Input file: " << file << std::endl;
    }
  }

  if (vm.count("port")) {
    int port = vm["port"].as<int>();
    cout << "port was set to " << port << ".\n";
    conf->setPort(port);
  } else {
    conf->setPort(8108);
  }

  if (vm.count("serverAddress")) {
    serverAddress = vm["serverAddress"].as<string>();
    conf->setServerAddress(serverAddress);
    cout << "server address was set to " << serverAddress << ".\n";
  } else {
    conf->setServerAddress("localhost");
  }

  if (vm.count("usePangea")) {
    conf->setUsePangea(vm["usePangea"].as<bool>());
    cout << "usePangea was set to " << conf->getUsePangea() << ".\n";
  } else {
    conf->setUsePangea(true);
  }

  if (vm.count("isMasterCatalogServer")) {
    bool isMasterCatalogServer = vm["isMasterCatalogServer"].as<bool>();
    conf->setMasterCatalogServer(isMasterCatalogServer);
    cout << "isMasterCatalogServer was set to " << isMasterCatalogServer
         << ".\n";
  } else {
    conf->setMasterCatalogServer(false);
  }

  string command;
  if (vm.count("command")) {
    command = vm["command"].as<string>();
    cout << "command was set to " << command << ".\n";
  } else {
    command = "0";
  }

  pdb::PDBClient pdbClient(conf->getPort(), conf->getServerAddress(),
                           make_shared<pdb::PDBLogger>("pdbClientLog"), false,
                           false);

  // start a storage client
  bool usePangea = conf->getUsePangea();

  cout << " Connected to IP: " << conf->getServerAddress() << endl;
  cout << " Using port: " << conf->getPort() << endl;
  cout << " Using pangea: " << conf->getUsePangea() << endl;

  pdb::StorageClient storageClient(conf->getPort(), conf->getServerAddress(),
                                   make_shared<pdb::PDBLogger>("clientLog"),
                                   true);

  string errMsg;

  if (command.compare("register-type") == 0) {
    cout << "***********Registering .so library" << endl;

    std::string soFile = vm["so-library-file"].as<std::string>();
    std::string soDataType = vm["library-type"].as<std::string>();

    std::cout << "Adding library: " << soFile << std::endl;

    // register the shared employee class
    if (!storageClient.registerType(soFile, errMsg)) {
      cout << "Not able to register type: " + errMsg << std::endl;
    } else {
      cout << "Registered type.\n";
    }
    cout << "Done.\n";

  } else if (command.compare("retrieve-type") == 0) {

    int typeId = vm["type-id"].as<int>();

    cout << "Retrieving typeId " << typeId << endl;

    // Local allocator
    const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 124};

    Handle<CatalogUserTypeMetadata> typeMetadata =
        makeObject<CatalogUserTypeMetadata>();
    string soFileObject = "temp.so";

    string soBytes;

    cout << "Done.\n";

  } else if (command.compare("register-node") == 0) {
    cout << "***********Registering node to cluster" << endl;

    std::string nodeIP = vm["node-ip"].as<std::string>();
    int nodePort = vm["node-port"].as<int>();
    std::string nodeName = vm["node-name"].as<std::string>();
    std::string nodeType = vm["node-type"].as<std::string>();
    int status = 0;

    if (!pdbClient.registerNode(nodeIP, nodePort, nodeName, nodeType, 1,
                                errMsg)) {
      std::cout << "Not able to register node metadata: " + errMsg << std::endl;
      std::cout << "Please change the parameters: nodeIP, port, nodeName, "
                   "nodeType, status."
                << std::endl;
    } else {
      std::cout << "Node metadata successfully added.\n";
    }
    cout << "Done.\n";

  } else if (command.compare("retrieve-node") == 0) {
    // Test to retrieve information about all nodes registered in the cluster

    std::string nodeName = vm["node-name"].as<std::string>();
    cout << "Done.\n";

  } else if (command.compare("register-db") == 0) {
    std::string databaseName = vm["db-name"].as<std::string>();

    if (!pdbClient.createDatabase(databaseName, errMsg)) {
      std::cout << "Not able to create database: " + errMsg << std::endl;
    } else {
      std::cout << "Database and its metadata successfully created.\n";
    }
    cout << "Done.\n";

  } else if (command.compare("remove-db") == 0) {
    std::string databaseName = vm["db-name"].as<std::string>();

    if (!pdbClient.removeDatabase (databaseName, errMsg)) {
        std :: cout << "Not able to remove database: " + errMsg <<
        std::endl;
    } else {
        std :: cout << "Database and its metadata successfully removed.\n";
    }
    cout << "Done.\n";

  } else if (command.compare("list-catalog") == 0) {
    // Test to retrieve a serialized version of the catalog

    cout << "Done.\n";

  } else if (command.compare("register-set") == 0) {

    std::string databaseName = vm["db-name"].as<std::string>();
    std::string typeName = vm["type-name"].as<std::string>();
    std::string setName = vm["set-name"].as<std::string>();

    cout << "***********ADD set to db" << endl;

    std::cout << "Adding set: " << setName << std::endl;
    std::cout << "     To db: " << databaseName << std::endl;
    std::cout << " With type: " << setName << std::endl;

    // now create a new set in that database
    if (typeName.compare("SharedEmployee") == 0) {
      if (!pdbClient.createSet(databaseName, setName, typeName, errMsg,
                               DEFAULT_PAGE_SIZE)) {
        std::cout << "Could not create set due to error: " + errMsg
                  << std::endl;
      } else {
        std::cout << "Set and its metadata successfully created.\n";
      }
    } else {
      std::cout << "Error type " << typeName << " is not registered.\n";
    }

    cout << "Done.\n";

  } else if (command.compare("remove-set") == 0) {

    std::string databaseName = vm["db-name"].as<std::string>();
    std::string typeName = vm["type-name"].as<std::string>();
    std::string setName = vm["set-name"].as<std::string>();

    cout << "***********Delete set from db" << endl;

    std::cout << "Deleting set: " << setName << std::endl;
    std::cout << "     From db: " << databaseName << std::endl;
    std::cout << " With type: " << setName << std::endl;

    // now the set from the database
    if (typeName.compare("SharedEmployee") == 0) {
      if (!pdbClient.removeSet(databaseName, setName, errMsg)) {
        std::cout << "Could not delete set due to error: " + errMsg
                  << std::endl;
      } else {
        std::cout << "Set and its metadata successfully deleted.\n";
      }
    } else {
    }

    cout << "Done.\n";

  } else if (command.compare("retrieve-set") == 0) {
    std::string setName = vm["set-name"].as<std::string>();
    std::string dbName = vm["db-name"].as<std::string>();

    cout << "Done.\n";

  } else if (command.compare("print-catalog") == 0) {
    //        std::string timeStamp = vm["timestamp"].as<std::string>();
 string toPrint = pdbClient.listUserDefinedTypes(errMsg);
        cout << "\n\n" << toPrint << endl;
    // creates an object to send request for printing all metadata
    pdb::Handle<pdb::CatalogPrintMetadata> printObject =
        pdb::makeObject<CatalogPrintMetadata>("", "0", "udts");

        toPrint = pdbClient.printCatalogMetadata(printObject, errMsg);
        cout << printObject->getMetadataToPrint().c_str() << endl;
        cout << "\n\n" << toPrint << endl;

        cout << "\n\nPrint all catalog " << endl;

printObject =
        pdb::makeObject<CatalogPrintMetadata>("", "0", "all");

        toPrint = pdbClient.listAllRegisteredMetadata(errMsg);
        cout << "\n\n" << toPrint << endl;
        
  }
}

#endif
