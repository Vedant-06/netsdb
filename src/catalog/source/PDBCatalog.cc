#include "PDBCatalog.h"

/* Specialization for adding Database metadata */
template <>
bool PDBCatalog::addItemToVector(Handle<CatalogDatabaseMetadata> &item,
                                 int &key) {
  auto begin = std::chrono::high_resolution_clock::now();
  registeredDatabasesMetadata->push_back(*item);
  registeredDatabases.insert(make_pair(item->getItemKey().c_str(), *item));
  auto end = std::chrono::high_resolution_clock::now();

  PDB_COUT << "----->Time Duration for DB addItemToVector:\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      end - begin)
                      .count())
           << " secs.\n"
           << endl;
  return true;
}

/* Specialization for adding Node metadata */
template <>
bool PDBCatalog::addItemToVector(Handle<CatalogNodeMetadata> &item, int &key) {
  registeredNodesMetadata->push_back(*item);
  registeredNodes.insert(make_pair(item->getItemKey().c_str(), *item));
  return true;
}

/* Specialization for adding Set metadata */
template <>
bool PDBCatalog::addItemToVector(Handle<CatalogSetMetadata> &item, int &key) {
  auto begin = std::chrono::high_resolution_clock::now();
  registeredSetsMetadata->push_back(*item);
  registeredSets.insert(make_pair(item->getItemKey().c_str(), *item));
  auto end = std::chrono::high_resolution_clock::now();

  PDB_COUT << "----->Time Duration for SET addItemToVector:\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      end - begin)
                      .count())
           << " secs." << endl;
  return true;
}

/* Specialization for adding User-defined type metadata */
template <>
bool PDBCatalog::addItemToVector(Handle<CatalogUserTypeMetadata> &item,
                                 int &key) {
  registeredUserDefinedTypesMetadata->push_back(*item);
  registeredUserDefinedTypes.insert(
      make_pair(item->getItemName().c_str(), *item));
  mapTypeNameToTypeID.insert(make_pair(item->getItemName().c_str(),
                                       atoi(item->getObjectID().c_str())));
  mapTypeIdToTypeName.insert(make_pair(atoi(item->getObjectID().c_str()),
                                       item->getItemName().c_str()));
  return true;
}

/* Specialization for updating Database metadata */
template <>
bool PDBCatalog::updateItemInVector(int &index,
                                    Handle<CatalogDatabaseMetadata> &item) {
  auto begin = std::chrono::high_resolution_clock::now();
  (*registeredDatabasesMetadata).assign(index, *item);
  registeredDatabases[(*item).getItemKey().c_str()] = (*item);
  auto end = std::chrono::high_resolution_clock::now();

  PDB_COUT << "----->Time Duration for DB updateItemInVector:\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      end - begin)
                      .count())
           << " secs." << endl;
  return true;
}

/* Specialization for updating Set metadata */
template <>
bool PDBCatalog::updateItemInVector(int &index,
                                    Handle<CatalogSetMetadata> &item) {
  auto begin = std::chrono::high_resolution_clock::now();
  (*registeredSetsMetadata).assign(index, *item);
  auto end = std::chrono::high_resolution_clock::now();
  PDB_COUT << "----->Time Duration for SET updateItemInVector:\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      end - begin)
                      .count())
           << " secs." << endl;
  return true;
}

/* Specialization for updating Node metadata */
template <>
bool PDBCatalog::updateItemInVector(int &index,
                                    Handle<CatalogNodeMetadata> &item) {
  (*registeredNodesMetadata).assign(index, *item);
  return true;
}

/* Specialization for updating User-defined type metadata */
template <>
bool PDBCatalog::updateItemInVector(int &index,
                                    Handle<CatalogUserTypeMetadata> &item) {
  (*registeredUserDefinedTypesMetadata).assign(index, *item);
  return true;
}

/* Specialization for deleting Set metadata */
template <>
bool PDBCatalog::deleteItemInVector(int &index,
                                    Handle<CatalogSetMetadata> &item) {
  Handle<Vector<CatalogSetMetadata>> tempContainter =
      makeObject<Vector<CatalogSetMetadata>>();
  for (int i = 0; i < (*registeredSetsMetadata).size(); i++) {
    if ((*item).getItemKey() != (*registeredSetsMetadata)[i].getItemKey()) {
      tempContainter->push_back((*registeredSetsMetadata)[i]);
    }
  }
  (*registeredSetsMetadata).clear();
  (*registeredSetsMetadata) = (*tempContainter);
  registeredSets.erase((*item).getItemKey().c_str());
  return true;
}

/* Specialization for deleting Node metadata */
template <>
bool PDBCatalog::deleteItemInVector(int &index,
                                    Handle<CatalogNodeMetadata> &item) {
  Handle<Vector<CatalogNodeMetadata>> tempContainter =
      makeObject<Vector<CatalogNodeMetadata>>();
  for (int i = 0; i < (*registeredNodesMetadata).size(); i++) {
    if ((*item).getItemKey() != (*registeredNodesMetadata)[i].getItemKey()) {
      tempContainter->push_back((*registeredNodesMetadata)[i]);
    }
  }
  (*registeredNodesMetadata).clear();
  (*registeredNodesMetadata) = (*tempContainter);
  registeredNodes.erase((*item).getItemKey().c_str());
  return true;
}

/* Specialization for deleting Database metadata */
template <>
bool PDBCatalog::deleteItemInVector<CatalogDatabaseMetadata>(
    int &index, Handle<CatalogDatabaseMetadata> &item) {
  Handle<Vector<CatalogDatabaseMetadata>> tempContainter =
      makeObject<Vector<CatalogDatabaseMetadata>>();
  for (int i = 0; i < (*registeredDatabasesMetadata).size(); i++) {
    if ((*item).getItemKey() !=
        (*registeredDatabasesMetadata)[i].getItemKey()) {
      tempContainter->push_back((*registeredDatabasesMetadata)[i]);
    }
  }
  (*registeredDatabasesMetadata).clear();
  (*registeredDatabasesMetadata) = (*tempContainter);
  registeredDatabases.erase((*item).getItemKey().c_str());
  return true;
}

/* Specialization for deleting User-defined type metadata */
template <>
bool PDBCatalog::deleteItemInVector(int &index,
                                    Handle<CatalogUserTypeMetadata> &item) {
  Handle<Vector<CatalogUserTypeMetadata>> tempContainter =
      makeObject<Vector<CatalogUserTypeMetadata>>();
  for (int i = 0; i < (*registeredUserDefinedTypesMetadata).size(); i++) {
    if ((*item).getItemKey() !=
        (*registeredUserDefinedTypesMetadata)[i].getItemKey()) {
      tempContainter->push_back((*registeredUserDefinedTypesMetadata)[i]);
    }
  }
  (*registeredUserDefinedTypesMetadata).clear();
  (*registeredUserDefinedTypesMetadata) = (*tempContainter);
  registeredUserDefinedTypes.erase((*item).getItemKey().c_str());
  return true;
}

void errorLogCallback(void *pArg, int iErrCode, const char *zMsg) {
  fprintf(stderr, "(%d) %s\n", iErrCode, zMsg);
}

PDBCatalog::PDBCatalog(PDBLoggerPtr logger, string location) {
  auto begin = std::chrono::high_resolution_clock::now();

  pdb::UseTemporaryAllocationBlock(1024 * 1024 * 128);

  sqlite3_config(SQLITE_CONFIG_LOG, errorLogCallback, NULL);
  pthread_mutex_init(&(registerMetadataMutex), NULL);
  this->logger = logger;

  // sets the paths for the location of the catalog files
  catalogRootPath = location + "/";
  string catalogPath = catalogRootPath + "pdbCatalog/";
  catalogFilename = catalogPath + "plinyCatalog.db";
  setUriPath("file:" + catalogFilename);
  tempPath = catalogRootPath + "tmp_so_files";

  // Creates the parent folder for the catalog
  // If location exists, only opens it.
  if (mkdir(catalogRootPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) ==
      -1) {
    PDB_COUT << "Parent catalog folder " << catalogRootPath
             << " was not created, it already exists." << std::endl;
  } else {
    PDB_COUT << "Parent catalog folder " << catalogRootPath
             << "  was created/opened." << std::endl;
  }

  // Creates a location folder for storing the sqlite file containing metadata
  // for this PDB instance.
  // If location exists, only opens it.
  if (mkdir(catalogPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
    PDB_COUT << "Catalog folder " << catalogPath
             << " was not created, it already exists." << std::endl;
  } else {
    PDB_COUT << "Catalog folder " << catalogPath << " was created/opened."
             << std::endl;
  }

  // creates temp folder for extracting so_files (only if folder doesn't exist)
  const int folder =
      mkdir(tempPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  if (folder == -1) {
    PDB_COUT << "Folder " << tempPath << " was not created, it already exists."
             << std::endl;
  } else {
    PDB_COUT << "Folder " << tempPath
             << " for temporary shared libraries was created/opened."
             << std::endl;
  }

  listUsersInCluster = makeObject<Vector<Handle<CatalogUserTypeMetadata>>>();
  registeredNodesMetadata = makeObject<Vector<CatalogNodeMetadata>>();
  registeredSetsMetadata = makeObject<Vector<CatalogSetMetadata>>();
  registeredDatabasesMetadata = makeObject<Vector<CatalogDatabaseMetadata>>();
  registeredUserDefinedTypesMetadata =
      makeObject<Vector<CatalogUserTypeMetadata>>();

  catalogContents = makeObject<Map<String, Handle<Vector<Object>>>>();

  // Populates a map to convert strings from PDBObject names to SQLite table
  // names in order to create query strings
  mapsPDBOjbect2SQLiteTable.insert(
      make_pair(PDBCatalogMsgType::CatalogPDBNode, "pdb_node"));
  mapsPDBOjbect2SQLiteTable.insert(
      make_pair(PDBCatalogMsgType::CatalogPDBDatabase, "pdb_database"));
  mapsPDBOjbect2SQLiteTable.insert(
      make_pair(PDBCatalogMsgType::CatalogPDBSet, "pdb_set"));
  mapsPDBOjbect2SQLiteTable.insert(
      make_pair(PDBCatalogMsgType::CatalogPDBUser, "pdb_user"));
  mapsPDBOjbect2SQLiteTable.insert(make_pair(
      PDBCatalogMsgType::CatalogPDBPermissions, "pdb_user_permission"));
  mapsPDBOjbect2SQLiteTable.insert(
      make_pair(PDBCatalogMsgType::CatalogPDBRegisteredObject, "data_types"));

  // Populates a map to convert strings from Vector<PDBOjbect> names to SQLite
  // table names in order to create query strings
  mapsPDBArrayOjbect2SQLiteTable.insert(
      make_pair(PDBCatalogMsgType::CatalogPDBNode, "pdb_node"));
  mapsPDBArrayOjbect2SQLiteTable.insert(
      make_pair(PDBCatalogMsgType::CatalogPDBDatabase, "pdb_database"));
  mapsPDBArrayOjbect2SQLiteTable.insert(
      make_pair(PDBCatalogMsgType::CatalogPDBSet, "pdb_set"));
  mapsPDBArrayOjbect2SQLiteTable.insert(
      make_pair(PDBCatalogMsgType::CatalogPDBUser, "pdb_user"));
  mapsPDBArrayOjbect2SQLiteTable.insert(make_pair(
      PDBCatalogMsgType::CatalogPDBPermissions, "pdb_user_permission"));
  mapsPDBArrayOjbect2SQLiteTable.insert(
      make_pair(PDBCatalogMsgType::CatalogPDBRegisteredObject, "data_types"));

  auto end = std::chrono::high_resolution_clock::now();
  this->logger->debug(
      "----->Time Duration for PDBCatalog constructor:\t " +
      std::to_string(
          std::chrono::duration_cast<std::chrono::duration<float>>(end - begin)
              .count()) +
      " secs.");
}

PDBCatalog::~PDBCatalog() {
  this->logger->debug("Catalog destructor called!!!");
  int deletedFiles = remove(tempPath.c_str());
  if (deletedFiles == -1) {
    PDB_COUT << " Error trying to remove temporary folder: " << tempPath
             << endl;
  } else {
    PDB_COUT << " Temporary folder: " << tempPath << " has been removed."
             << endl;
  }
  sqlite3_close_v2(sqliteDBHandler);
  pthread_mutex_destroy(&(registerMetadataMutex));
}

PDBLoggerPtr PDBCatalog::getLogger() { return this->logger; }

void PDBCatalog::closeSQLiteHandler() {
  this->logger->debug("Closing SQLite Handler!!!");
  sqlite3_close_v2(sqliteDBHandler);
}

void PDBCatalog::setCatalogVersion(string version) {
  catalogVersion = version;
}

string PDBCatalog::getCatalogVersion() {
  return catalogVersion;
}

bool PDBCatalog::getSerializedCatalog(
  string fileName,
  string version,
  string &returnedBytes,
  string &errorMessage) {

  errorMessage = "";

  string fullName = catalogRootPath + "plinyCatalog.db";

  fstream file(fullName.c_str(), ios::in | ios::binary);
  if (!file) {
    errorMessage = "The file " + fullName + " was not found\n";
  }

  file.seekp(0, fstream::end);
  streampos numBytes = file.tellp();
  file.seekp(0, fstream::beg);

  char *buffer = new char[numBytes];
  file.read(buffer, numBytes);
  returnedBytes = string(buffer, numBytes);
  return true;
}

void PDBCatalog::open() {
  auto begin = std::chrono::high_resolution_clock::now();
  auto query = begin;
  auto load = begin;

  sqliteDBHandler = NULL;
  int ret = 0;
  // If database doesn't exist creates database along with tables, otherwise,
  // opens database without creating a database/tables.
  std::cout << "catalog server uri: " << uriPath << std::endl;
  if ((ret = sqlite3_open_v2(uriPath.c_str(), &sqliteDBHandler,
                             SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE |
                                 SQLITE_OPEN_URI | SQLITE_OPEN_FULLMUTEX,
                             NULL)) == SQLITE_OK) {

    query = std::chrono::high_resolution_clock::now();
    // These two SQLite flags optimize insertions/deletions/updates to the
    // tables by buffering data prior to writing to disk
    sqlite3_exec(sqliteDBHandler, "PRAGMA synchronous=OFF", NULL, NULL, NULL);
    sqlite3_exec(sqliteDBHandler, "PRAGMA journal_mode=memory", NULL, NULL,
                 NULL);

    // Create tables if they don't exist, they are created with primary key to
    // prevent duplicates
    catalogSqlQuery(
        "CREATE TABLE IF NOT EXISTS data_types (itemID TEXT PRIMARY KEY, "
        "itemInfo BLOB, soBytes BLOB, timeStamp INTEGER);");
    catalogSqlQuery(
        "CREATE TABLE IF NOT EXISTS metrics (itemID TEXT PRIMARY KEY, "
        "itemInfo BLOB, soBytes BLOB, timeStamp INTEGER);");
    catalogSqlQuery(
        "CREATE TABLE IF NOT EXISTS pdb_node (itemID TEXT PRIMARY KEY, "
        "itemInfo BLOB, timeStamp INTEGER);");
    catalogSqlQuery(
        "CREATE TABLE IF NOT EXISTS pdb_database (itemID TEXT PRIMARY KEY, "
        "itemInfo BLOB, timeStamp INTEGER);");
    catalogSqlQuery(
        "CREATE TABLE IF NOT EXISTS pdb_set (itemID TEXT PRIMARY KEY, "
        "itemInfo BLOB, timeStamp INTEGER);");
    catalogSqlQuery(
        "CREATE TABLE IF NOT EXISTS pdb_user (itemID TEXT PRIMARY KEY, "
        "itemInfo BLOB, timeStamp INTEGER);");
    catalogSqlQuery("CREATE TABLE IF NOT EXISTS pdb_user_permission (itemID "
                    "TEXT PRIMARY KEY, "
                    "itemInfo BLOB, timeStamp INTEGER);");

    load = std::chrono::high_resolution_clock::now();

    // Loads into memory all metadata so the CatalogServer can access them
    loadsMetadataIntoMemory();

    PDB_COUT << "Catalog database successfully open." << endl;

  } else {
    PDB_COUT << "Error opening catalog database." << endl;
  }

  auto end = std::chrono::high_resolution_clock::now();
  PDB_COUT << "Time Duration for open SQLite:\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      query - begin)
                      .count())
           << " secs." << endl;
  PDB_COUT << "Time Duration for sqlite CREATES/OPENS:\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      load - query)
                      .count())
           << " secs." << endl;
  PDB_COUT << "Time Duration for loads in memory includes selects in SQLite:\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(end -
                                                                           load)
                      .count())
           << " secs." << endl;
  PDB_COUT << "Time Duration for open:\t "
           << std::to_string(
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      end - begin)
                      .count())
           << " secs." << endl;
}

void PDBCatalog::loadsMetadataIntoMemory() {
  string errorMessage;

  // an empty string lists all entries in a given category
  string emptyString("");

  // retrieves metadata from the SQLite DB and populates containers
  if (getMetadataFromCatalog(false,
                             emptyString,
                             registeredNodesMetadata,
                             errorMessage,
                             PDBCatalogMsgType::CatalogPDBNode) == false)

      this->logger->debug(errorMessage);

  (*catalogContents)[String("nodes")] =
      unsafeCast<Vector<Object>>(registeredNodesMetadata);

  if (getMetadataFromCatalog(false,
                             emptyString,
                             registeredSetsMetadata,
                             errorMessage,
                             PDBCatalogMsgType::CatalogPDBSet) == false)

      this->logger->debug(errorMessage);

  (*catalogContents)[String("sets")] =
      unsafeCast<Vector<Object>>(registeredSetsMetadata);

  if (getMetadataFromCatalog(false,
                             emptyString,
                             registeredDatabasesMetadata,
                             errorMessage,
                             PDBCatalogMsgType::CatalogPDBDatabase) == false)

      this->logger->debug(errorMessage);

  (*catalogContents)[String("dbs")] =
      unsafeCast<Vector<Object>>(registeredDatabasesMetadata);

  if (getMetadataFromCatalog(false,
                             emptyString,
                             registeredUserDefinedTypesMetadata,
                             errorMessage,
                             PDBCatalogMsgType::CatalogPDBRegisteredObject) == true) {

    (*catalogContents)[String("udfs")] =
        unsafeCast<Vector<Object>>(registeredUserDefinedTypesMetadata);

    // populates maps
    // retrieves databases metadata
    for (int i = 0; i < (*registeredDatabasesMetadata).size(); i++) {
      PDB_COUT << "Retrieving db "
               << string((*registeredDatabasesMetadata)[i].getItemKey())
               << " | "
               << string((*registeredDatabasesMetadata)[i].getItemName())
               << endl;

      registeredDatabases.insert(
          make_pair((*registeredDatabasesMetadata)[i].getItemName().c_str(),
                    (*registeredDatabasesMetadata)[i]));
    }

    // retrieves sets metadata
    for (int i = 0; i < (*registeredSetsMetadata).size(); i++) {
      PDB_COUT << "Retrieving set "
               << string((*registeredSetsMetadata)[i].getItemKey())
               << " | "
               << string((*registeredSetsMetadata)[i].getItemName())
               << endl;

      registeredSets.insert(
          make_pair((*registeredSetsMetadata)[i].getItemKey().c_str(),
                    (*registeredSetsMetadata)[i]));
    }

    // retrieves nodes metadata
    for (int i = 0; i < (*registeredNodesMetadata).size(); i++) {
      PDB_COUT << "Retrieving node "
               << string((*registeredNodesMetadata)[i].getItemKey())
               << " | "
               << string((*registeredNodesMetadata)[i].getNodeIP())
               << endl;

       registeredNodes.insert(
          make_pair((*registeredNodesMetadata)[i].getItemKey().c_str(),
                    (*registeredNodesMetadata)[i]));
    }

    // retrieves user-defined types metadata
    for (int i = 0; i < (*registeredUserDefinedTypesMetadata).size(); i++) {
      PDB_COUT << "Retrieving node "
               << string((*registeredUserDefinedTypesMetadata)[i].getObjectID())
               << " | "
               << string((*registeredUserDefinedTypesMetadata)[i].getItemName())
               << endl;

      registeredUserDefinedTypes.insert(make_pair(
          (*registeredUserDefinedTypesMetadata)[i].getItemName().c_str(),
          (*registeredUserDefinedTypesMetadata)[i]));
    }

  } else {
    PDB_COUT << errorMessage << endl;
  }

  cout << "=========================================" << endl;
  cout << "Metadata Registered in the PDB Catalog " << endl;
  cout << "-----------------------------------------" << endl;

  string metadataToPrint;
  printsAllCatalogMetadata(metadataToPrint, errorMessage);
  cout << metadataToPrint.c_str() << endl;

  cout << "\nAll Metadata retrieved and loaded." << endl;
  cout << "=========================================" << endl;
}

void PDBCatalog::getModifiedMetadata(
    Handle<CatalogPrintMetadata> &itemMetadata) {

  string errorMessage;
  string dateAsString = itemMetadata->getTimeStamp().c_str();

  Handle<Vector<CatalogNodeMetadata>> _registeredNodesMetadata =
      makeObject<Vector<CatalogNodeMetadata>>();
  Handle<Vector<CatalogSetMetadata>> _registeredSetsMetadata =
      makeObject<Vector<CatalogSetMetadata>>();
  Handle<Vector<CatalogDatabaseMetadata>> _registeredDatabasesMetadata =
      makeObject<Vector<CatalogDatabaseMetadata>>();
  Handle<Vector<CatalogUserTypeMetadata>> _registeredUserDefinedTypesMetadata =
      makeObject<Vector<CatalogUserTypeMetadata>>();

  // retrieves metadata from the sqlite DB and populates containers
  if (getMetadataFromCatalog(true, dateAsString, _registeredNodesMetadata,
                             errorMessage,
                             PDBCatalogMsgType::CatalogPDBNode) == false)
    this->logger->debug(errorMessage);

  if (getMetadataFromCatalog(true, dateAsString, _registeredSetsMetadata,
                             errorMessage,
                             PDBCatalogMsgType::CatalogPDBSet) == false)
    this->logger->debug(errorMessage);

  if (getMetadataFromCatalog(true, dateAsString, _registeredDatabasesMetadata,
                             errorMessage,
                             PDBCatalogMsgType::CatalogPDBDatabase) == false)
    this->logger->debug(errorMessage);

  if (getMetadataFromCatalog(
          true, dateAsString, _registeredUserDefinedTypesMetadata, errorMessage,
          PDBCatalogMsgType::CatalogPDBRegisteredObject) == true) {

    for (int i = 0; i < (*_registeredUserDefinedTypesMetadata).size(); i++) {
      PDB_COUT
          << "RETRIEVING TYPE "
          << string((*_registeredUserDefinedTypesMetadata)[i].getItemName())
          << " | "
          << string((*_registeredUserDefinedTypesMetadata)[i].getObjectID())
          << endl;

      registeredUserDefinedTypes.insert(make_pair(
          (*_registeredUserDefinedTypesMetadata)[i].getItemName().c_str(),
          (*_registeredUserDefinedTypesMetadata)[i]));
    }

  } else {
    PDB_COUT << errorMessage << endl;
  }

  cout << "=========================================" << endl;
  cout << "PDB Metadata Registered in the Catalog " << endl;

  cout << "\nNodes in cluster: " +
              std::to_string((int)(*_registeredNodesMetadata).size())
       << endl;
  cout << "----------------------------" << endl;

  for (int i = 0; i < (*_registeredNodesMetadata).size(); i++) {
    cout << (*_registeredNodesMetadata)[i].printShort() << endl;
  }

  cout << "\nDatabases: " +
              std::to_string((int)(*_registeredDatabasesMetadata).size())
       << endl;
  cout << "----------------------------" << endl;

  for (int i = 0; i < (*_registeredDatabasesMetadata).size(); i++) {
    cout << (*_registeredDatabasesMetadata)[i].printShort() << endl;
  }
  cout << "\nSets: " + std::to_string((int)(*_registeredSetsMetadata).size())
       << endl;
  cout << "----------------------------" << endl;

  for (int i = 0; i < (*_registeredSetsMetadata).size(); i++) {
    cout << (*_registeredSetsMetadata)[i].printShort() << endl;
  }
  cout << "\nUser-defined types: " +
              std::to_string((int)(*_registeredUserDefinedTypesMetadata).size())
       << endl;
  cout << "----------------------------" << endl;

  for (int i = 0; i < (*_registeredUserDefinedTypesMetadata).size(); i++) {
    cout << (*_registeredUserDefinedTypesMetadata)[i].printShort() << endl;
  }
  cout << "\nAll Metadata properly retrieved and\n"
          "loaded into memory!"
       << endl;
  cout << "--------------------------------------" << endl;
}

void PDBCatalog::printsAllCatalogMetadata(std::string &outputString, std::string &errMsg) {

    listNodesInCluster(outputString, errMsg);
    listRegisteredDatabases(outputString, errMsg);
    listUserDefinedTypes(outputString, errMsg);

}

/* Lists the Nodes registered in the catalog. */
void PDBCatalog::listNodesInCluster(std::string &outputString,
                                    std::string &errMsg) {

    outputString += "\nI. Nodes in Cluster ("
                 +  std::to_string((int)(*registeredNodesMetadata).size())
                 +  ")\n";

  for (int i = 0; i < (*registeredNodesMetadata).size(); i++) {
    outputString += (*registeredNodesMetadata)[i].printShort() + "\n";
  }

}

/* Lists the Databases registered in the catalog. */
void PDBCatalog::listRegisteredDatabases(std::string &outputString,
                                         std::string &errMsg) {

  outputString += "\nII. Databases ("
               +  std::to_string((int)(*registeredDatabasesMetadata).size())
               +  ")\n";

  for (int i = 0; i < (*registeredDatabasesMetadata).size(); i++) {
    outputString += (*registeredDatabasesMetadata)[i].printShort() + "\n";
  }

}

/* Lists the Sets for a given database registered in the catalog. */
void PDBCatalog::listRegisteredSetsForADatabase(std::string databaseName,
                                                std::string &outputString,
                                                std::string &errMsg) {

  outputString += "\nSets ("
               +  std::to_string((int)(*registeredSetsMetadata).size())
               +  ")\n";

  for (int i = 0; i < (*registeredSetsMetadata).size(); i++) {
    outputString += (*registeredSetsMetadata)[i].printShort() + "\n";
  }

}

/* Lists the user-defined types registered in the catalog. */
void PDBCatalog::listUserDefinedTypes(std::string &outputString,
                                      std::string &errMsg) {

    outputString += "\nIII. User-defined types ("
                 +  std::to_string((int)(*registeredUserDefinedTypesMetadata).size())
                 +  ")\n";

  for (int i = 0; i < (*registeredUserDefinedTypesMetadata).size(); i++) {
    outputString +=
        (*registeredUserDefinedTypesMetadata)[i].printShort() + "\n";
  }

}

template <class CatalogMetadataType>
bool PDBCatalog::getMetadataFromCatalog(
    bool onlyModified, string key,
    Handle<pdb::Vector<CatalogMetadataType>> &returnedItems,
    string &errorMessage, int metadataCategory) {

  pdb::String emptyString("");

  pthread_mutex_lock(&(registerMetadataMutex));

  sqlite3_stmt *statement = NULL;

  string queryString = "SELECT itemID, itemInfo, timeStamp from " +
                       mapsPDBArrayOjbect2SQLiteTable[metadataCategory];

  // if empty string then retrieves all items in the table, otherwise only the
  // item with the given key
  if (onlyModified == true)
    queryString.append(" where timeStamp > ").append(key).append("");
  else if (key != "")
    queryString.append(" where itemID = '").append(key).append("'");

  PDB_COUT << queryString << endl;

  this->logger->debug(queryString);
  if (sqlite3_prepare_v2(sqliteDBHandler, queryString.c_str(), -1, &statement,
                         NULL) == SQLITE_OK) {

    int res = 0;
    while (1) {
      res = sqlite3_step(statement);

      if (res == SQLITE_ROW) {
        // retrieves the serialized record
        int numBytes = sqlite3_column_bytes(statement, 1);

        PDB_COUT << "entry " << sqlite3_column_text(statement, 0)
                 << " timestamp " << sqlite3_column_int(statement, 2) << endl;

        Record<CatalogMetadataType> *recordBytes =
            (Record<CatalogMetadataType> *)malloc(numBytes);

        memcpy(recordBytes, sqlite3_column_blob(statement, 1), numBytes);

        // get the object
        Handle<CatalogMetadataType> returnedObject =
            recordBytes->getRootObject();

        string itemId = returnedObject->getItemKey().c_str();
        this->logger->debug("itemId=" + itemId);
        returnedItems->push_back(*returnedObject);
        free(recordBytes);

      } else if (res == SQLITE_DONE) {
        break;
      }
    }
    sqlite3_finalize(statement);
    pthread_mutex_unlock(&(registerMetadataMutex));

    return true;
  } else {

    string error = sqlite3_errmsg(sqliteDBHandler);

    if (error != "not an error") {
      this->logger->writeLn((string)queryString + " " + error);
    }

    sqlite3_finalize(statement);

    pthread_mutex_unlock(&(registerMetadataMutex));
    return false;
  }
}

bool PDBCatalog::registerUserDefinedObject(
  int16_t typeCode,
  pdb::Handle<CatalogUserTypeMetadata> &objectToRegister,
  const string &objectBytes,
  const string &typeName,
  const string &fileName,
  const string &tableName,
  string &errorMessage) {

  bool isSuccess = false;

  pthread_mutex_lock(&(registerMetadataMutex));
  PDB_COUT << "inside registerUserDefinedObject\nobjectBytes "
           << std::to_string(objectBytes.size()) << endl;
  PDB_COUT << "typeName " << typeName << endl;
  PDB_COUT << "fileName " << fileName << endl;
  PDB_COUT << "tableName " << tableName << endl;

  bool success = false;
  errorMessage = "";

  sqlite3_stmt *stmt = NULL;
  uint8_t *serializedBytes = NULL;

  string queryString("");
  queryString = "INSERT INTO " + tableName +
                " (itemID, itemInfo, soBytes, timeStamp) "
                "VALUES(?, ?, ?, strftime('%s', 'now', 'localtime'))";

  PDB_COUT << "QueryString = " << queryString << endl;

  int rc =
      sqlite3_prepare_v2(sqliteDBHandler, queryString.c_str(), -1, &stmt, NULL);
  if (rc != SQLITE_OK) {
    errorMessage = "Prepared statement failed. " +
                   (string)sqlite3_errmsg(sqliteDBHandler) + "\n";
    PDB_COUT << "errorMessage" << errorMessage << endl;

    success = false;
  } else {
    PDB_COUT << "Pass else" << endl;
    // Gets the number of registered objects in the catalog
    int totalRegisteredTypes = (int)registeredUserDefinedTypes.size();

    PDB_COUT << "ASSIGN TYPE_ID-> " << std::to_string(typeCode) << " for Type "
             << typeName << endl;

    string newObjectId = std::to_string(typeCode);
    String newObjectIndex = String(std::to_string(typeCode));

    PDB_COUT << "   Object Id -----> " << newObjectId;
    String idToRegister = String(newObjectId);
    String tableToRegister = String(tableName);
    String typeToRegister = String(typeName);

    // Sets object ID prior to serialization
    objectToRegister->setItemId(newObjectIndex);
    objectToRegister->setObjectId(idToRegister);
    objectToRegister->setItemKey(typeToRegister);
    objectToRegister->setItemName(typeToRegister);
    String empty = String(" ");
    objectToRegister->setLibraryBytes(empty);

    // gets the raw bytes of the object
    Record<CatalogUserTypeMetadata> *metadataBytes =
        getRecord<CatalogUserTypeMetadata>(objectToRegister);

    // TODO I might be able to directly save the bytes into sqlite without the
    // memcpy
    serializedBytes = (uint8_t *)malloc(metadataBytes->numBytes());
    memcpy(serializedBytes, metadataBytes, metadataBytes->numBytes());
    size_t soBytesSize = objectBytes.length();
    this->logger->debug("size soBytesSize  " + std::to_string(soBytesSize));

    rc = sqlite3_bind_text(stmt, 1, typeToRegister.c_str(), -1, SQLITE_STATIC);
    rc = sqlite3_bind_blob(stmt, 2, serializedBytes, metadataBytes->numBytes(),
                           SQLITE_STATIC);
    rc = sqlite3_bind_blob(stmt, 3, objectBytes.c_str(), soBytesSize,
                           SQLITE_STATIC);

    // Inserts newly added object into containers
    addItemToVector(objectToRegister, totalRegisteredTypes);

    if (rc != SQLITE_OK) {
      errorMessage = "Bind operation failed. " +
                     (string)sqlite3_errmsg(sqliteDBHandler) + "\n";
      this->logger->debug(errorMessage);
      success = false;
    } else {
      rc = sqlite3_step(stmt);
      if (rc != SQLITE_DONE) {
        if (sqlite3_errcode(sqliteDBHandler) == SQLITE_CONSTRAINT) {
          success = false;
          errorMessage =
              fileName + " is already registered in the catalog!" + "\n";
        } else
          errorMessage = "Query execution failed. " +
                         (string)sqlite3_errmsg(sqliteDBHandler) + "\n";
      } else {
        registeredUserDefinedTypes.insert(
            make_pair(typeName, *objectToRegister));

        success = true;
        this->logger->writeLn(
            "Dynamic library successfully stored in catalog!");
      }
    }
    sqlite3_finalize(stmt);
    free(serializedBytes);
  }
  PDB_COUT << errorMessage << std::endl;

  pthread_mutex_unlock(&(registerMetadataMutex));
  return isSuccess;
}

map<string, CatalogUserTypeMetadata> PDBCatalog::getUserDefinedTypesList() {
  return registeredUserDefinedTypes;
}

// Retrieves a Shared Library file stored as BLOB in SQLite
// and writes it into a temporary folder/file so it can be loaded using
// dlopen.
bool PDBCatalog::retrievesDynamicLibrary(
    string itemId, string tableName,
    Handle<CatalogUserTypeMetadata> &returnedItem, string &returnedSoLibrary,
    string &errorMessage) {

  pthread_mutex_lock(&(registerMetadataMutex));

  sqlite3_blob *pBlob = NULL;
  sqlite3_stmt *pStmt = NULL;

  errorMessage = "";

  string query = "SELECT itemID, itemInfo, soBytes FROM " + tableName +
                 " where itemID = ?;";
  PDB_COUT << "query: " << query << " " << itemId << endl;
  if (sqlite3_prepare_v2(sqliteDBHandler, query.c_str(), -1, &pStmt, NULL) !=
      SQLITE_OK) {
    errorMessage = "Error query not well formed: " +
                   (string)sqlite3_errmsg(sqliteDBHandler) + "\n";

    sqlite3_reset(pStmt);
    PDB_COUT << errorMessage << endl;
    pthread_mutex_unlock(&registerMetadataMutex);
    return false;
  }

  sqlite3_bind_text(pStmt, 1, itemId.c_str(), -1, SQLITE_STATIC);

  if (sqlite3_step(pStmt) != SQLITE_ROW) {
    errorMessage = "Error item not found in database: " +
                   (string)sqlite3_errmsg(sqliteDBHandler) + "\n";
    sqlite3_reset(pStmt);
    PDB_COUT << errorMessage << endl;

    pthread_mutex_unlock(&registerMetadataMutex);
    return false;
  }

  // retrieves metadata stored as serialized pdb :: Object
  int numBytes = sqlite3_column_bytes(pStmt, 1);
  Record<CatalogUserTypeMetadata> *recordBytes =
      (Record<CatalogUserTypeMetadata> *)malloc(numBytes);

  if (recordBytes == nullptr) {
    PDB_COUT << "FATAL ERROR: Out of memory!" << std::endl;

    pthread_mutex_unlock(&registerMetadataMutex);
    exit(-1);
  }

  memcpy(recordBytes, sqlite3_column_blob(pStmt, 1), numBytes);
  Handle<CatalogUserTypeMetadata> returnedObject = recordBytes->getRootObject();
  returnedItem =
      deepCopyToCurrentAllocationBlock<CatalogUserTypeMetadata>(returnedObject);

  if (returnedItem == nullptr) {
    PDB_COUT << "FATAL ERROR: Corrupted CatalogUserTypeMetadata!" << std::endl;

    pthread_mutex_unlock(&registerMetadataMutex);
    free(recordBytes);
    return false;
  }

  PDB_COUT << "Metadata created for item " << string(returnedItem->getItemId())
           << endl;
  PDB_COUT << "Metadata created for item " << string(returnedItem->getItemKey())
           << endl;
  PDB_COUT << "file size= " + std::to_string(numBytes) << endl;

  // retrieves the bytes for the .so library
  numBytes = sqlite3_column_bytes(pStmt, 2);

  char *buffer = new char[numBytes];
  memcpy(buffer, sqlite3_column_blob(pStmt, 2), numBytes);
  returnedSoLibrary = string(buffer, numBytes);
  delete[] buffer;

  PDB_COUT << "buffer bytes size " + std::to_string(returnedSoLibrary.size())
           << endl;

  sqlite3_reset(pStmt);
  sqlite3_blob_close(pBlob);

  pthread_mutex_unlock(&(registerMetadataMutex));
  free(recordBytes);
  return true;
}

void PDBCatalog::deleteTempSoFiles(string filePath) {
  struct dirent *next_file = NULL;
  DIR *theFolder = NULL;

  theFolder = opendir(filePath.c_str());
  if (theFolder != NULL) {
    while ((next_file = readdir(theFolder))) {
      if (strcmp(next_file->d_name, ".") == 0 ||
          strcmp(next_file->d_name, "..") == 0) {
        continue;
      }
      string fullName = filePath + "/" + next_file->d_name;
      remove(fullName.c_str());
    }
  }
}

template <class CatalogMetadataType>
bool PDBCatalog::addMetadataToCatalog(
    pdb::Handle<CatalogMetadataType> &metadataValue,
    int &metadataCategory,
    string &errorMessage) {

  auto begin = std::chrono::high_resolution_clock::now();
  auto realSQLInsert = begin;
  pthread_mutex_lock(&(registerMetadataMutex));

  Handle<CatalogMetadataType> metadataObject =
      makeObject<CatalogMetadataType>();

  bool isSuccess = false;
  sqlite3_stmt *stmt = NULL;

  string sqlStatement = "INSERT INTO " +
                        mapsPDBOjbect2SQLiteTable[metadataCategory] +
                        " (itemID, itemInfo, timeStamp) VALUES (?, ?, "
                        "strftime('%s', 'now', 'localtime'))";

  // gets the size of the container for a given type of metadata and
  // uses it to assign the index of a metadata item in its container
  int newId = getLastId(metadataCategory);
  pdb::String newKeyValue = String(std::to_string(newId));
  string metadataKey = metadataValue->getItemKey().c_str();
  metadataValue->setItemId(newKeyValue);

  auto metadataBytes = getRecord<CatalogMetadataType>(metadataValue);
  size_t numberOfBytes = metadataBytes->numBytes();

  auto prepareRecord = std::chrono::high_resolution_clock::now();

  this->logger->debug(sqlStatement + " with key= " + metadataKey);
  // Prepares statement
  if ((sqlite3_prepare_v2(sqliteDBHandler, sqlStatement.c_str(), -1, &stmt,
                          NULL)) != SQLITE_OK) {

    errorMessage =
        "Prepared statement failed. " + (string)sqlite3_errmsg(sqliteDBHandler);
    this->logger->writeLn(errorMessage);
    isSuccess = false;
  }

  // Binds key for this piece of metadata
  if ((sqlite3_bind_text(stmt, 1, metadataKey.c_str(), -1, SQLITE_STATIC)) !=
      SQLITE_OK) {
    errorMessage = "Bind operation failed. " +
                   (string)sqlite3_errmsg(sqliteDBHandler) + "\n";
    isSuccess = false;
  }

  // Binds value for this piece of metadata (as a pdb serialized set of bytes)
  if ((sqlite3_bind_blob(stmt, 2, metadataBytes, numberOfBytes,
                         SQLITE_STATIC)) != SQLITE_OK) {
    errorMessage = "Bind operation failed. " +
                   (string)sqlite3_errmsg(sqliteDBHandler) + "\n";
    isSuccess = false;
  }

  // Runs the insert statement
  if (catalogSqlStep(stmt, errorMessage)) {
    // Metadata item inserted in sqlite then add to pdb :: Vector  in memory

    realSQLInsert = std::chrono::high_resolution_clock::now();

    addItemToVector(metadataValue, metadataCategory);
    isSuccess = true;

  } else {
    errorMessage = "Cannot add new item to Catalog";
    this->logger->writeLn(errorMessage);
  }

  sqlite3_finalize(stmt);

  pthread_mutex_unlock(&(registerMetadataMutex));

  if (isSuccess == true) {
    this->logger->debug("The following item metadata was stored in SQLite and "
                        "loaded into Catalog memory:");
    this->logger->debug(metadataValue->printShort());
  }
  this->logger->writeLn(errorMessage);
  auto end = std::chrono::high_resolution_clock::now();
  this->logger->debug(
      "Time Duration for Prepare record\t" +
      std::to_string(std::chrono::duration_cast<std::chrono::duration<float>>(
                         prepareRecord - begin)
                         .count()) +
      " secs.");
  this->logger->debug(
      "Time Duration for Real INSERT INTO to sqlite:\t " +
      std::to_string(std::chrono::duration_cast<std::chrono::duration<float>>(
                         realSQLInsert - prepareRecord)
                         .count()) +
      " secs.");
  this->logger->debug(
      "----->Time Duration for addMetadataToCatalog Total\t " +
      std::to_string(std::chrono::duration_cast<std::chrono::duration<float>>(
                         end - realSQLInsert)
                         .count()) +
      " secs.");
  return isSuccess;
}

template <class CatalogMetadataType>
bool PDBCatalog::updateMetadataInCatalog(
    pdb::Handle<CatalogMetadataType> &metadataValue, int &metadataCategory,
    string &errorMessage) {
  auto begin = std::chrono::high_resolution_clock::now();
  auto prepareRecord = begin;
  auto realSQLInsert = begin;

  pthread_mutex_lock(&(registerMetadataMutex));

  // gets the key and index for this item in order to update the sqlite table
  // and
  // update the container in memory
  String metadataKey = metadataValue->getItemKey();
  int metadataIndex = std::atoi(metadataValue->getItemId().c_str());

  bool isSuccess = false;
  sqlite3_stmt *stmt = NULL;
  string sqlStatement = "UPDATE " +
                        mapsPDBOjbect2SQLiteTable[metadataCategory] +
                        " set itemInfo =  ?, timeStamp = strftime('%s', 'now', "
                        "'localtime') where itemId = ?";

  this->logger->debug(sqlStatement + " id: " + metadataKey.c_str());

  auto metadataBytes = getRecord<CatalogMetadataType>(metadataValue);

  size_t numberOfBytes = metadataBytes->numBytes();

  prepareRecord = std::chrono::high_resolution_clock::now();
  // Prepares statement
  if ((sqlite3_prepare_v2(sqliteDBHandler, sqlStatement.c_str(), -1, &stmt,
                          NULL)) != SQLITE_OK) {
    errorMessage =
        "Prepared statement failed. " + (string)sqlite3_errmsg(sqliteDBHandler);
    this->logger->writeLn(errorMessage);
    isSuccess = false;
  }

  // Binds value for this piece of metadata (as a pdb serialized set of bytes)
  if ((sqlite3_bind_blob(stmt, 1, metadataBytes, numberOfBytes,
                         SQLITE_STATIC)) != SQLITE_OK) {

    errorMessage = "Bind operation failed. " +
                   (string)sqlite3_errmsg(sqliteDBHandler) + "\n";
    isSuccess = false;
  }

  // Binds key for this piece of metadata
  if ((sqlite3_bind_text(stmt, 2, metadataKey.c_str(), -1, SQLITE_STATIC)) !=
      SQLITE_OK) {
    errorMessage = "Bind operation failed. " +
                   (string)sqlite3_errmsg(sqliteDBHandler) + "\n";
    isSuccess = false;
  }

  PDB_COUT << errorMessage << endl;

  // Runs the update statement
  if (catalogSqlStep(stmt, errorMessage)) {
    // if sqlite update goes well, updates container
    realSQLInsert = std::chrono::high_resolution_clock::now();

    updateItemInVector(metadataIndex, metadataValue);
    isSuccess = true;

    PDB_COUT << "Metadata for the following item has been updated:"
             << metadataValue->printShort() << endl;
  } else {
    errorMessage = "Cannot update item in Catalog";
  }
  PDB_COUT << errorMessage << endl;

  sqlite3_finalize(stmt);

  pthread_mutex_unlock(&(registerMetadataMutex));
  auto end = std::chrono::high_resolution_clock::now();
  this->logger->debug(
      "Time Duration for Prepare record\t" +
      std::to_string(std::chrono::duration_cast<std::chrono::duration<float>>(
                         prepareRecord - begin)
                         .count()) +
      " secs.");
  this->logger->debug(
      "Time Duration for Real UPDATE INTO to sqlite:\t " +
      std::to_string(std::chrono::duration_cast<std::chrono::duration<float>>(
                         realSQLInsert - prepareRecord)
                         .count()) +
      " secs.");
  this->logger->debug(
      "----->Time Duration for updateMetadataInCatalog Total\t " +
      std::to_string(std::chrono::duration_cast<std::chrono::duration<float>>(
                         end - realSQLInsert)
                         .count()) +
      " secs.");
  return isSuccess;
}

template <class CatalogMetadataType>
bool PDBCatalog::deleteMetadataInCatalog(
    pdb::Handle<CatalogMetadataType> metadataValue, int &metadataCategory,
    string &errorMessage) {

  pthread_mutex_lock(&(registerMetadataMutex));
  // gets the key and index for this item in order to update the sqlite table
  // and update the container in memory
  String metadataKey = metadataValue->getItemKey();
  int metadataIndex = std::atoi(metadataValue->getItemId().c_str());

  bool isSuccess = false;
  sqlite3_stmt *stmt = NULL;
  string sqlStatement = "DELETE from " +
                        mapsPDBOjbect2SQLiteTable[metadataCategory] +
                        " where itemId = ?";

  PDB_COUT << sqlStatement << " id: " << metadataKey.c_str() << endl;

  // Prepares statement
  if ((sqlite3_prepare_v2(sqliteDBHandler, sqlStatement.c_str(), -1, &stmt,
                          NULL)) != SQLITE_OK) {
    errorMessage =
        "Prepared statement failed. " + (string)sqlite3_errmsg(sqliteDBHandler);
    PDB_COUT << errorMessage << endl;
    isSuccess = false;
  }

  // Binds key for this piece of metadata
  if ((sqlite3_bind_text(stmt, 1, metadataKey.c_str(), -1, SQLITE_STATIC)) !=
      SQLITE_OK) {
    errorMessage = "Bind operation failed. " +
                   (string)sqlite3_errmsg(sqliteDBHandler) + "\n";
    isSuccess = false;
  }

  PDB_COUT << errorMessage << endl;

  // Runs the update statement
  if (catalogSqlStep(stmt, errorMessage)) {
    // if sqlite update goes well, updates container
    deleteItemInVector(metadataIndex, metadataValue);
    isSuccess = true;
  } else {
    errorMessage = "Cannot delete item in Catalog";
  }
  PDB_COUT << errorMessage << endl;

  sqlite3_finalize(stmt);

  PDB_COUT << "Updating " << (*metadataValue).printShort() << endl;

  pthread_mutex_unlock(&(registerMetadataMutex));
  return isSuccess;
}

map<string, CatalogNodeMetadata> PDBCatalog::getListOfNodesInCluster() {
  return registeredNodes;
}

string PDBCatalog::getMapsPDBOjbect2SQLiteTable(int typeOfObject) {
  return mapsPDBOjbect2SQLiteTable[typeOfObject];
}

int PDBCatalog::getLastId(int &metadataCategory) {
  int lastId = 0;
  switch (metadataCategory) {

  case PDBCatalogMsgType::CatalogPDBNode: {
    lastId = registeredNodesMetadata->size();
    break;
  }

  case PDBCatalogMsgType::CatalogPDBDatabase: {
    lastId = registeredDatabasesMetadata->size();
    break;
  }

  case PDBCatalogMsgType::CatalogPDBSet: {
    lastId = registeredSetsMetadata->size();
    break;
  }
  }
  return lastId;
}

std::ostream &operator<<(std::ostream &out, PDBCatalog &catalog) {
  out << "--------------------------" << endl;
  out << "PDB Metadata Registered in the Catalog: " << endl;

  out << "\n   Number of cluster nodes registered: " +
             std::to_string((int)catalog.registeredNodesMetadata->size())
      << endl;
  for (int i = 0; i < catalog.registeredNodesMetadata->size(); i++) {
    out << "      Id: "
        << (*catalog.registeredNodesMetadata)[i].getItemId().c_str()
        << " | Node name: "
        << (*catalog.registeredNodesMetadata)[i].getItemName().c_str()
        << " | Node Address: "
        << (*catalog.registeredNodesMetadata)[i].getItemKey().c_str() << ":"
        << (*catalog.registeredNodesMetadata)[i].getNodePort() << endl;
  }

  out << "\n   Number of databases registered: " +
             std::to_string((int)catalog.registeredDatabasesMetadata->size())
      << endl;
  for (int i = 0; i < catalog.registeredDatabasesMetadata->size(); i++) {
    out << "      Id: "
        << (*catalog.registeredDatabasesMetadata)[i].getItemId().c_str()
        << " | Database: "
        << (*catalog.registeredDatabasesMetadata)[i].getItemName().c_str()
        << endl;
  }

  out << "\n   Number of sets registered: " +
             std::to_string((int)catalog.registeredSetsMetadata->size())
      << endl;
  for (int i = 0; i < catalog.registeredSetsMetadata->size(); i++) {
    out << "      Id: "
        << (*catalog.registeredSetsMetadata)[i].getItemId().c_str()
        << " | Key: "
        << (*catalog.registeredSetsMetadata)[i].getItemKey().c_str()
        << " | Database: "
        << (*catalog.registeredSetsMetadata)[i].getDBName().c_str()
        << " | Set: "
        << (*catalog.registeredSetsMetadata)[i].getItemName().c_str() << endl;
  }

  out << "\n   Number of users registered: " +
             std::to_string((int)catalog.listUsersInCluster->size())
      << endl;
  for (int i = 0; i < catalog.listUsersInCluster->size(); i++) {
    out << (*catalog.listUsersInCluster)[i]->getItemName() << endl;
  }

  out << "\n   Number of user-defined types registered: " +
             std::to_string(
                 (int)catalog.registeredUserDefinedTypesMetadata->size())
      << endl;
  for (int i = 0; i < catalog.registeredUserDefinedTypesMetadata->size(); i++) {
    out << "      Id: "
        << (*catalog.registeredUserDefinedTypesMetadata)[i].getItemId().c_str()
        << " | Type Name: "
        << (*catalog.registeredUserDefinedTypesMetadata)[i]
               .getItemName()
               .c_str()
        << endl;
  }
  out << "--------------------------" << endl;
  return out;
}

string PDBCatalog::itemName2ItemId(int &metadataCategory, string &key) {
  switch (metadataCategory) {
  case PDBCatalogMsgType::CatalogPDBNode: {
    return registeredNodes[key].getItemKey().c_str();
    break;
  }

  case PDBCatalogMsgType::CatalogPDBDatabase: {
    return registeredDatabases[key].getItemId().c_str();
    break;
  }

  case PDBCatalogMsgType::CatalogPDBSet: {
    return registeredSets[key].getItemKey().c_str();
    break;
  }

  case PDBCatalogMsgType::CatalogPDBRegisteredObject: {
    // User-defined types are stored in a different type of map
    return registeredUserDefinedTypes[key].getObjectID().c_str();
    break;
  }

  default: {
    return "Unknown request!";
    break;
  }
  }
}

bool PDBCatalog::keyIsFound(int &metadataCategory, string &key, string &value) {
  value = "";

  switch (metadataCategory) {
  case PDBCatalogMsgType::CatalogPDBNode: {
    auto p = registeredNodes.find(key);
    if (p != registeredNodes.end()) {
      value = p->second.getItemKey().c_str();
      return true;
    }
    break;
  }

  case PDBCatalogMsgType::CatalogPDBDatabase: {
    auto p = registeredDatabases.find(key);
    if (p != registeredDatabases.end()) {
      value = p->second.getItemKey().c_str();
      return true;
    }
    break;
  }

  case PDBCatalogMsgType::CatalogPDBSet: {
    auto p = registeredSets.find(key);
    if (p != registeredSets.end()) {
      value = p->second.getItemKey().c_str();
      return true;
    }
    break;
  }

  case PDBCatalogMsgType::CatalogPDBRegisteredObject: {
    auto p = registeredUserDefinedTypes.find(key);
    if (p != registeredUserDefinedTypes.end()) {
      value = p->second.getItemKey().c_str();
      return true;
    }
    break;
  }

  default: {
    return false;
    break;
  }
  }

  return false;
}

// Executes a sqlite3 query on the catalog database given by a query string.
bool PDBCatalog::catalogSqlQuery(string queryString) {

  sqlite3_stmt *statement = NULL;

  if (sqlite3_prepare_v2(sqliteDBHandler, queryString.c_str(), -1, &statement,
                         NULL) == SQLITE_OK) {
    int result = 0;
    result = sqlite3_step(statement);
    sqlite3_finalize(statement);
    return true;
  } else {
    string error = sqlite3_errmsg(sqliteDBHandler);
    if (error != "not an error") {
      this->logger->writeLn((string)queryString + " " + error);
      return true;
    } else {
      return false;
    }
  }
}

// Executes a sqlite3 statement query on the catalog database given by a query
// string
// works for inserts, updates and deletes
bool PDBCatalog::catalogSqlStep(sqlite3_stmt *stmt, string &errorMsg) {

  int rc = 0;
  if ((rc = sqlite3_step(stmt)) == SQLITE_DONE) {
    return true;
  } else {
    errorMsg = sqlite3_errmsg(sqliteDBHandler);
    if (errorMsg != "not an error") {
      this->logger->writeLn("Problem running sqlite statement: " + errorMsg);
      return true;
    } else {
      return false;
    }
  }
}

void PDBCatalog::getListOfDatabases(
    Handle<Vector<CatalogDatabaseMetadata>> &databasesInCatalog,
    const string &keyToSearch) {

  String searchForKey(keyToSearch);
  this->logger->debug("keyToSearch=" + keyToSearch);
  this->logger->debug("searchForKey=" + string(searchForKey));
  if (keyToSearch == "")
    databasesInCatalog = registeredDatabasesMetadata;
  else {
    for (int i = 0; i < (*registeredDatabasesMetadata).size(); i++) {
      this->logger->debug("i=" + std::to_string(i));
      this->logger->debug((*registeredDatabasesMetadata)[i].getItemKey());
      if (searchForKey == (*registeredDatabasesMetadata)[i].getItemKey()) {
        databasesInCatalog->push_back((*registeredDatabasesMetadata)[i]);
      }
    }
  }
}

void PDBCatalog::getListOfSets(
    Handle<Vector<CatalogSetMetadata>> &setsInCatalog,
    const string &keyToSearch) {
  String searchForKey(keyToSearch);
  if (keyToSearch == "")
    setsInCatalog = registeredSetsMetadata;
  else {
    for (int i = 0; i < (*registeredSetsMetadata).size(); i++) {
      if (searchForKey == (*registeredSetsMetadata)[i].getItemKey()) {
        setsInCatalog->push_back((*registeredSetsMetadata)[i]);
      }
    }
  }
}

void PDBCatalog::getListOfNodes(
    Handle<Vector<CatalogNodeMetadata>> &nodesInCatalog,
    const string &keyToSearch) {
  String searchForKey(keyToSearch);
  if (keyToSearch == "")
    nodesInCatalog = registeredNodesMetadata;
  else {
    for (int i = 0; i < (*registeredNodesMetadata).size(); i++) {
      if (searchForKey == (*registeredNodesMetadata)[i].getItemKey()) {
        nodesInCatalog->push_back((*registeredNodesMetadata)[i]);
      }
    }
  }
}

/* Explicit instantiation for adding Node Metadata to the catalog */
template bool PDBCatalog::addMetadataToCatalog(
    pdb::Handle<CatalogNodeMetadata> &metadataValue,
    int &catalogType,
    string &errorMessage);

/* Explicit instantiation for adding Set Metadata to the catalog */
template bool
PDBCatalog::addMetadataToCatalog(
    pdb::Handle<CatalogSetMetadata> &metadataValue,
    int &catalogType, 
    string &errorMessage);

/* Explicit instantiation for adding Database Metadata to the catalog */
template bool PDBCatalog::addMetadataToCatalog(
    pdb::Handle<CatalogDatabaseMetadata> &metadataValue,
    int &catalogType,
    string &errorMessage);

/* Explicit instantiation for updating Node Metadata in the catalog */
template bool PDBCatalog::updateMetadataInCatalog(
    pdb::Handle<CatalogNodeMetadata> &metadataValue, int &catalogType,
    string &errorMessage);

/* Explicit instantiation for updating Set Metadata in the catalog */
template bool PDBCatalog::updateMetadataInCatalog(
    pdb::Handle<CatalogSetMetadata> &metadataValue, int &catalogType,
    string &errorMessage);

/* Explicit instantiation for updating Database Metadata in the catalog */
template bool PDBCatalog::updateMetadataInCatalog(
    pdb::Handle<CatalogDatabaseMetadata> &metadataValue, int &catalogType,
    string &errorMessage);

/* Explicit instantiation for updating User-defined Type Metadata in the
 * catalog
 */
template bool PDBCatalog::updateMetadataInCatalog(
    pdb::Handle<CatalogUserTypeMetadata> &metadataValue, int &catalogType,
    string &errorMessage);

/* Explicit instantiation for deleting Node Metadata from the catalog */
template bool PDBCatalog::deleteMetadataInCatalog(
    pdb::Handle<CatalogNodeMetadata> metadataValue, int &catalogType,
    string &errorMessage);

/* Explicit instantiation for deleting Set Metadata from the catalog */
template bool PDBCatalog::deleteMetadataInCatalog(
    pdb::Handle<CatalogSetMetadata> metadataValue, int &catalogType,
    string &errorMessage);

/* Explicit instantiation for deleting Database Metadata from the catalog */
template bool PDBCatalog::deleteMetadataInCatalog(
    pdb::Handle<CatalogDatabaseMetadata> metadataValue, int &catalogType,
    string &errorMessage);

/* Explicit instantiation for deleting User-defined Type Metadata from the
 * catalog */
template bool PDBCatalog::deleteMetadataInCatalog(
    pdb::Handle<CatalogUserTypeMetadata> metadataValue, int &catalogType,
    string &errorMessage);
