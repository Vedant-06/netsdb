#ifndef PDB_CATALOG_H_
#define PDB_CATALOG_H_

#include <algorithm>
#include <chrono>
#include <ctime>
#include <dirent.h>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <map>
#include <pthread.h>
#include <sqlite3.h>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "CatalogDatabaseMetadata.h"
#include "CatalogNodeMetadata.h"
#include "CatalogPermissionsMetadata.h"
#include "CatalogPrintMetadata.h"
#include "CatalogSetMetadata.h"
#include "CatalogStandardDatabaseMetadata.h"
#include "CatalogStandardNodeMetadata.h"
#include "CatalogStandardSetMetadata.h"
#include "CatalogStandardUserTypeMetadata.h"
#include "CatalogUserTypeMetadata.h"
#include "CatalogUserTypeMetadata.h"
#include "Handle.h"
#include "InterfaceFunctions.h"
#include "Object.h"
#include "PDBCatalogMsgType.h"
#include "PDBCommunicator.h"
#include "PDBDebug.h"
#include "PDBLogger.h"

using namespace std;
using namespace pdb;

/**
 * PDBCatalog encapsulates the storage of metadata for an instance of PlinyCompute.
 * The underlying persistent storage is an embedded SQLite database with the
 * following tables:
 *
 *   - data_types: user-defined types stored as shared libraries
 *   - pdb_node: metadata about nodes in a cluster
 *   - pdb_database: metadata about databases
 *   - pdb_set: metadata about sets in databases
 *   - pdb_user: metadata about users in a PDB instance (to be implemented)
 *   - pdb_user_permission: access to databases for users of PDB (to be
 *     implemented)
 *
 * This class also offers containers for loading these metadata in memory upon
 * launching an instance of CatalogServer, allowing clients to access them.
 */

class PDBCatalog;
typedef shared_ptr<PDBCatalog> PDBCatalogPtr;

class PDBCatalog {
public:
  /**
   * Constructor
   *
   * @param logger the catalog logger
   * @param location the path to the location of the catalog (is relative to the
   * pdb root directory
   *
   */
  PDBCatalog(PDBLoggerPtr logger, string location);

  /**
   * Destructor, deletes temporary files.
   *
   */
  ~PDBCatalog();

  /**
   * Opens an instance of the catalog, retrieves metadata from the sqlite
   * instance and invokes a method for loading retrieved metadata into memory.
   *
   */
  void open();

  /**
   * registers a user-defined type in the Catalog
   * @typeCode is the typeID already assigned
   * @param objectToRegister contains the metadata for this object
   *
   * @param objectBytes are the binary bytes of the Shared Library encapsulated
   *        as an string
   * @param typeName is a string obtained from the shared library, which
   *        identifies the type (e.g. myCoolType)
   * @param fileName is the path+filename of the Shared Library in the local
   *        file system (e.g. mypath/mylibs/libMyCoolType.so
   * @param tableName is a string identifying the type of the object (e.g.
   *        metrics, data_type)
   * @return true on success
   */
  bool registerUserDefinedObject(
      int16_t typeCode,
      pdb::Handle<CatalogUserTypeMetadata> &objectToRegister,
      const string &objectBytes,
      const string &typeName,
      const string &fileName,
      const string &tableName,
      string &errorMessage);

  /**
   * addMetadataToCatalog registers a Metadata item into the Catalog, basically
   * saving it as serialized bytes in an Sqlite table, it then adds that
   * metadata item into memory.
   *
   * @param metadataValue encapsulates the object with the metadata
   * @param metadataCategory identifies the metadata category (values are
   *        defined in PDBCatalogMsgType)
   * @param errorMessage error message
   * @return true on success
   */
  template <class CatalogMetadataType>
  bool addMetadataToCatalog(
      Handle<CatalogMetadataType> &metadataValue,
      int &metadataCategory,
      string &errorMessage);

  /**
   * loads a registered Metadata item into memory so it can be
   * accessed by the CatalogServer
   *
   * @param item encapsulates the object with the metadata
   * @param key is a string key used for maps
   * @return true on success
   */
  template <class CatalogMetadataType>
  bool addItemToVector(Handle<CatalogMetadataType> &item, int &key);

  /**
   * updates an existing Metadata item in Sqlite along with its content in
   * memory
   *
   * @param metadataValue encapsulates the object with the metadata
   * @param metadataCategory identifies the metadata category (values are
   *        defined in PDBCatalogMsgType)
   * @param errorMessage error message
   * @return true on success
   */
  template <class CatalogMetadataType>
  bool updateMetadataInCatalog(
      pdb::Handle<CatalogMetadataType> &metadataValue,
      int &metadataCategory,
      string &errorMessage);

  /**
   * deletes an existing Metadata item in Sqlite along
   * with its content in memory
   *
   * @param metadataValue encapsulates the object with the metadata
   * @param metadataCategory identifies the metadata category (values are
   * defined in PDBCatalogMsgType)
   * @param errorMessage error message
   * @return true on success
   */
  template <class CatalogMetadataType>
  bool deleteMetadataInCatalog(
      pdb::Handle<CatalogMetadataType> metadataValue,
      int &metadataCategory,
      string &errorMessage);

  /**
   * updates a registered Metadata item in memory so the
   * changes are visible to the CatalogServer
   *
   * @param index is the position in the container
   * @param item is the new metadata content
   * @return true on success
   */
  template <class CatalogMetadataType>
  bool updateItemInVector(
       int &index,
       Handle<CatalogMetadataType> &item);

  /**
   * deletes a registered Metadata item in memory so the
   * changes are visible to the CatalogServer
   *
   * @param index is the position in the container
   * @param item is the new metadata content
   * @return true on success
   */
  template <class CatalogMetadataType>
  bool deleteItemInVector(
       int &index,
       Handle<CatalogMetadataType> &item);

  /**
   * returns the list of nodes in the cluster registered in the catalog
   */
  map<string, CatalogNodeMetadata> getListOfNodesInCluster();

  /**
   * retrieves the bytes of the entire catalog, this could be used
   * it one wants to ship the catalog from the master node to a different
   * machine
   *
   * @param fileName is the name of the catalog file
   * @param version is the version to retrieve
   * @param returnedBytes contains the bytes encapsulated in a string
   * @param errorMessage the error returned
   *
   * @return true on success
   */
  bool getSerializedCatalog(string fileName, string version,
                            string &returnedBytes, string &errorMessage);

  /**
   * sets the version of the catalog, this is typically called
   * when updates are made to the catalog
   *
   * @param version contains the signature of the version
   */
  void setCatalogVersion(string version);

  /**
   * gets the version of the catalog as a string
   */
  string getCatalogVersion();

  /**
   * gets a version of the content of the catalog as a printable object
   */
  void getModifiedMetadata(Handle<CatalogPrintMetadata> &itemMetadata);

  /**
   * retrieves all Metadata from Sqlite for a given category and returns it
   * in a container
   *
   * @param onlyModified, if true will return only entries that were created
   * after a given timestamp, which is passed with the "key" parameter
   * @param key if blank returns all items in the category, otherwise, only the
   * one matching the key (in the case of timestamp the format is
   * Timestamp in milliseconds, for example January 1, 2014 12:00:00 AM
   * is passed as: 1388534400000) and will return all entries created after that
   * given timestamp
   * @param returnedEntries is a Vector of the Objects
   * @param errorMessage error message
   * @param metadataCategory identifies the metadata category (values are
   * defined in PDBCatalogMsgType)
   * @return true on success
   */
  template <class CatalogMetadataType>
  bool getMetadataFromCatalog(
      bool onlyModified,
      string key,
      Handle<pdb::Vector<CatalogMetadataType>> &returnedEntries,
      string &errorMessage,
      int metadataCategory);

  /**
   * gets the number of items in a given Metadata category, so the id
   * can be properly set prior to the addition of a new item.
   *
   * @param metadataCategory is an enum to identify the category of metadata
   * (values are defined in PDBCatalogMsgType)
   * @return the number of items as an int
   */
  int getLastId(int &metadataCategory);

  /**
   * maps the id of a Metadata item given its name
   *
   * @param metadataCategory is an enum to identify the category of metadata
   * (values are defined in PDBCatalogMsgType)
   * @return the value of the item
   */
  string itemName2ItemId(int &metadataCategory, string &key);

  /**
   * checks if a Metadata item is found in its container
   *
   * @param metadataCategory is an enum to identify the category of metadata
   * (values are defined in PDBCatalogMsgType)
   * @param key to search for
   * @return the value of the item
   */
  bool keyIsFound(int &metadataCategory, string &key, string &value);

  /**
   * Overloads the << operator so other classes can print the content of the
   * catalog metadata
   */
  friend std::ostream &operator<<(std::ostream &out, PDBCatalog &catalog);

  /**
   * Gets the logger for the Catalog
   */
  PDBLoggerPtr getLogger();

  /**
   * Returns a map with metadata for all user-defined types registered in the
   * catalog
   */
  map<string, CatalogUserTypeMetadata> getUserDefinedTypesList();

  /**
   * Prints all metadata registered in the catalog
   */
  void printsAllCatalogMetadata(
       std::string &outputString,
       std::string &errMsg);

  /**
   * Lists the Databases registered in the catalog.
   */
  void listRegisteredDatabases(
       std::string &outputString,
       std::string &errMsg);

  /**
   * Lists the Sets for a given database registered in the catalog.
   */
  void listRegisteredSetsForADatabase(
       std::string databaseName,
       std::string &outputString,
       std::string &errMsg);

  /**
   * Lists the Nodes registered in the catalog.
   */
  void listNodesInCluster(
       std::string &outputString,
       std::string &errMsg);

  /**
   * Lists the user-defined types registered in the catalog.
   */
  void listUserDefinedTypes(
       std::string &outputString,
       std::string &errMsg);

  /**
   * Retrieves a dynamic library stored as BLOB in the Catalog
   * returns 1 if success, 0 otherwise.
   */
  bool retrievesDynamicLibrary(
       string fileName,
       string tableName,
       Handle<CatalogUserTypeMetadata> &returnedItem,
       string &returnedSoLibrary,
       string &errorName);

  /**
   * returns the metadata for all Databases registered in the catalog.
   * The information is retrieved from memory rather than disk (SQLite).
   *
   * @param key to search for
   * @return a Vector of items
   */
  void getListOfDatabases(
      Handle<Vector<CatalogDatabaseMetadata>> &databasesInCatalog,
      const string &key);

  /**
   * returns the metadata for all Sets registered in the catalog.
   * The information is retrieved from memory rather than disk (SQLite).
   *
   * @param key to search for
   * @return a Vector of items
   */
  void getListOfSets(
       Handle<Vector<CatalogSetMetadata>> &setsInCatalog,
       const string &key);

  /**
   * returns the metadata for all Nodes registered in the catalog.
   * The information is retrieved from memory rather than disk
   * (SQLite).
   *
   * @param key to search for
   * @return a Vector of items
   */
  void getListOfNodes(Handle<Vector<CatalogNodeMetadata>> &nodesInCatalog,
                      const string &key);

  /**
   * closes the SQLite DB handler once is no longer needed
   */
  void closeSQLiteHandler();

private:
  /**
   * returns the name of a Sqlite table for a given category of metadata
   */
  string getMapsPDBOjbect2SQLiteTable(int typeOfObject);

  /**
   * loads all Metadata stored in Sqlite and populates the corresponding containers
   */
  void loadsMetadataIntoMemory();

  /**
   * Mutex for ensuring proper synchronization of catalog metadata access
   */
  pthread_mutex_t registerMetadataMutex;

  /**
   * A string that indicates the version of this catalog instance
   */
  string catalogVersion;

  /**
   * The logger for this catalog instance
   */
  PDBLoggerPtr logger;

  /**
   * Handler for the SQLite instance, which stores all catalog metadata
   */
  sqlite3 *sqliteDBHandler = NULL;

  /**
   * List unique metadata entries
   */
  vector<CatalogDatabaseMetadata> dbList;

  /**
   * Contains information of users registered in the catalog.
   * (To be implemented)
   */
  Handle<pdb::Vector<Handle<CatalogUserTypeMetadata>>> listUsersInCluster;

  /**
   * Map of users, given the name of a user as a string,
   * lists all databases belonging to that user. (To be implemented)
   */
  multimap<string, CatalogUserTypeMetadata> mapUsersInCluster;

  /**
   * Maps a typeName to its typeID for a user-defined type registered in the
   * catalog
   */
  map<string, int16_t> mapTypeNameToTypeID;

  /**
   * Maps a typeID to its typeName for a user-defined type registered in the
   * catalog
   */
  map<int16_t, string> mapTypeIdToTypeName;

  /**
   * Maps a node IP:Port to its Metadata
   **/
  map<string, CatalogNodeMetadata> registeredNodes;

  /**
   * Container for keeping in memory metadata for all registered nodes
   * in the catalog
   **/
  Handle<Vector<CatalogNodeMetadata>> registeredNodesMetadata;

  /**
   * Maps a set ID to its Metadata
   */
  map<string, CatalogSetMetadata> registeredSets;

  /**
   * Container for keeping in memory metadata for all registered sets
   * in the catalog
   */
  Handle<Vector<CatalogSetMetadata>> registeredSetsMetadata;

  /**
   * Maps a database name to its Metadata
   */
  map<string, CatalogDatabaseMetadata> registeredDatabases;

  /**
   * Container for keeping in memory metadata for all registered databases
   * in the catalog
   */
  Handle<Vector<CatalogDatabaseMetadata>> registeredDatabasesMetadata;

  /**
   * Maps a user-defined type to its Metadata
   */
  map<string, CatalogUserTypeMetadata> registeredUserDefinedTypes;

  /**
   * Container for keeping in memory metadata for all registered user-defined
   * types in the catalog
   */
  Handle<Vector<CatalogUserTypeMetadata>> registeredUserDefinedTypesMetadata;

  /**
   * Container for keeping in memory  all the metadata registered in
   * the catalog. Used when a remote catalog (i.e. non-master catalog)
   * requests an update "pull" of newly registered metadata.
   */
  Handle<Map<String, Handle<Vector<Object>>>> catalogContents;

  /**
   * URI string that represents the location of the plinyCatalog.db instance
   * so it can be use for opening connections to the SQLite catalog file
   **/
  string uriPath;

  /**
   * Root path where the catalog resides, relative to where PlinyCompute runs
   */
  string catalogRootPath;

  /**
   * String that contains the full name of the catalog file, to be used
   * in SQLite statements
   */
  string catalogFilename;

  /**
   * String that represents the temporary location where shared library files
   * will be stored at runtime
   */
  string tempPath;

  /**
   * Creates a statement from a string and executes the query, returning
   * true if successful
   */
  bool catalogSqlQuery(string statement);

  /**
   * Executes an sql statement in sqlite3 (insert, update or delete)
   */
  bool catalogSqlStep(sqlite3_stmt *stmt, string &errorMsg);

  /**
   * Creates a temporary folder to place the shared library files, returns 0 if
   * success
   */
  int createsTempPath();

  /**
   * Sets the URI path where the SQLite database is located
   */
  void setUriPath(string thePath) { uriPath = thePath; }

  /**
   * Deletes all shared library files from the temp directory (is called by the
   * destructor)
   */
  void deleteTempSoFiles(string filePath);

  /**
   * Maps the name of an SQLite table given the type of metadata it contains.
   * Used for composing prepared SQL statements in SQLite, given a type Metadata
   */
  map<int, string> mapsPDBOjbect2SQLiteTable;

  /**
   * Maps the name of an SQLite table given the type of metadata it contains.
   * Used for composing prepared SQL statements in SQLite, given a tyep of
   * Metadata in a container
   */
  map<int, string> mapsPDBArrayOjbect2SQLiteTable;
};

#endif /* PDB_CATALOG_H_ */
