/*
 * CatalogNodeMetadata.h
 *
 *  Created on: Sept 12, 2016
 *      Author: carlos
 */

#ifndef CATALOG_NODE_METADATA_H_
#define CATALOG_NODE_METADATA_H_

#include <iostream>
#include "Object.h"
#include "PDBString.h"
#include "PDBVector.h"

//  PRELOAD %CatalogNodeMetadata%

using namespace std;

namespace pdb {

// This class serves to store information about a node in a cluster of PDB.
// It also provides methods for maintaining their associated metadata.
// Clients of this class will access this information using a handler to the catalog.

class CatalogNodeMetadata : public Object {
public:
    CatalogNodeMetadata() {}

    CatalogNodeMetadata(pdb::String nodeIdValue,
                        pdb::String nodeIPValue,
                        int nodePortValue,
                        pdb::String nodeNameValue,
                        pdb::String nodeTypeValue,
                        int nodeStatusValue)
        : nodeId(nodeIdValue),
          nodeIP(nodeIPValue),
          nodePort(nodePortValue),
          nodeName(nodeNameValue),
          nodeType(nodeTypeValue),
          nodeStatus(nodeStatusValue) {
        nodeAddress = std::string(nodeIP) + std::string(":") + std::to_string(nodePort);
    }

    // Copy constructor
    CatalogNodeMetadata(const CatalogNodeMetadata& pdbNodeToCopy) {
        nodeId = pdbNodeToCopy.nodeId;
        nodeIP = pdbNodeToCopy.nodeIP;
        nodePort = pdbNodeToCopy.nodePort;
        nodeName = pdbNodeToCopy.nodeName;
        nodeType = pdbNodeToCopy.nodeType;
        nodeStatus = pdbNodeToCopy.nodeStatus;
        nodeAddress = std::string(nodeIP) + std::string(":") + std::to_string(nodePort);
    }

    // Copy constructor
    CatalogNodeMetadata(const Handle<CatalogNodeMetadata>& pdbNodeToCopy) {
        nodeId = pdbNodeToCopy->getItemId();
        nodeIP = pdbNodeToCopy->getNodeIP();
        nodePort = pdbNodeToCopy->getNodePort();
        nodeName = pdbNodeToCopy->getItemName();
        nodeType = pdbNodeToCopy->getNodeType();
        nodeStatus = pdbNodeToCopy->getNodeStatus();
        nodeAddress = std::string(nodeIP) + std::string(":") + std::to_string(nodePort);
    }

    ~CatalogNodeMetadata() {}

    void setValues(pdb::String nodeIdValue,
                   pdb::String nodeIPValue,
                   int nodePortValue,
                   pdb::String nodeNameValue,
                   pdb::String nodeTypeValue,
                   int nodeStatusValue) {
        nodeId = nodeIdValue;
        nodeIP = nodeIPValue;
        nodePort = nodePortValue;
        nodeName = nodeNameValue;
        nodeType = nodeTypeValue;
        nodeStatus = nodeStatusValue;
        nodeAddress = std::string(nodeIP) + std::string(":") + std::to_string(nodePort);
    }

    pdb::String getItemKey() {
        return nodeAddress;
    }

    pdb::String getNodeIP() {
        return nodeIP;
    }

    pdb::String getItemId() {
        return nodeId;
    }

    pdb::String getItemName() {
        return nodeName;
    }

    pdb::String getNodeType() {
        return nodeType;
    }

    int getNodePort() {
        return nodePort;
    }

    int getNodeStatus() {
        return nodeStatus;
    }

    void setItemKey(pdb::String& itemKeyIn) {
        nodeAddress = itemKeyIn;
    }

    void setItemId(pdb::String& itemIdIn) {
        nodeId = itemIdIn;
    }


    void setItemIP(pdb::String& itemIPIn) {
        nodeIP = itemIPIn;
    }

    void setItemName(pdb::String& itemNameIn) {
        nodeName = itemNameIn;
    }

    void setNodePort(int& portIn) {
        nodePort = portIn;
    }

    string printShort() {
        string output;
        output = "   Node address: ";
        output.append(getItemKey().c_str())
            .append(" | ")
            .append(getItemName().c_str())
            .append(" | ")
            .append(getNodeType().c_str());
        return output;
    }


    friend std::ostream& operator<<(std::ostream& out, CatalogNodeMetadata& node) {
        out << "\nCluster Node Metadata" << endl;
        out << "-------------------" << endl;
        out << "       Node Id: " << node.getItemId().c_str() << endl;
        out << "       Node IP: " << node.getNodeIP().c_str() << endl;
        out << "     Node Port: " << node.getNodePort() << endl;
        out << "     Node Name: " << node.getItemName().c_str() << endl;
        out << "     Node Type: " << node.getNodeType().c_str() << endl;
        out << "-------------------\n" << endl;
        return out;
    }

    ENABLE_DEEP_COPY

private:
    pdb::String nodeId;
    pdb::String nodeIP;
    int nodePort;
    pdb::String nodeName;
    pdb::String nodeType;
    int nodeStatus;
    pdb::String nodeAddress;
};

} /* namespace pdb */

#endif /* CATALOG_NODE_METADATA_H_ */
