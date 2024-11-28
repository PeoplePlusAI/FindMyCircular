import json
import os

from py2neo import Graph, Node, NodeMatcher, Relationship

# Define Neo4j database connection parameters
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "yourpassword"


def create_knowledge_graph(json_directory):
    """
    Create a knowledge graph from JSON files in the specified directory.

    Args:
        json_directory (str): The directory containing JSON files to be processed.
    """
    # Connect to the Neo4j database
    graph = Graph(URI, auth=(USER, PASSWORD))

    # Delete all existing nodes and relationships in the graph
    graph.delete_all()

    # Create a unique constraint on the 'name' property of GovernmentDocument nodes
    graph.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:GovernmentDocument) REQUIRE d.name IS UNIQUE"
    )

    # Initialize a NodeMatcher for finding existing nodes
    matcher = NodeMatcher(graph)

    # Iterate over all JSON files in the specified directory
    for filename in os.listdir(json_directory):
        if filename.endswith(".json"):
            file_path = os.path.join(json_directory, filename)
            with open(file_path, "r") as file:
                data = json.load(file)

            # Create or merge the main document node
            main_doc = Node(
                "GovernmentDocument",
                name=data["name"],
                date_of_issue=data["date_of_issue"],
                summary=data["summary"],
                questions=data["questions"],
            )
            graph.merge(main_doc, "GovernmentDocument", "name")

            # Iterate over related documents and their relationship types
            for related_doc, relation_type in data["relations"].items():
                # Check if the related document node already exists
                existing_node = matcher.match(
                    "GovernmentDocument", name=related_doc
                ).first()

                if existing_node:
                    related_node = existing_node
                else:
                    # Create a new node for the related document if it doesn't exist
                    related_node = Node("GovernmentDocument", name=related_doc)
                    graph.create(related_node)

                # Create or merge the relationship between the main document and the related document
                relation = Relationship(main_doc, relation_type.upper(), related_node)
                graph.merge(relation)

            print(f"Processed {data['name']}")

    print("Knowledge graph creation complete.")


# Call the function to create the knowledge graph from JSON files in the specified directory
create_knowledge_graph("legalAgent/langraphKG/output")
