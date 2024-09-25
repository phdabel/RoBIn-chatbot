import os
import logging
import pandas as pd
from retry import retry
from neo4j import GraphDatabase

# Paths to CSV files containing Cochrane data
BIAS_TYPES_CSV_PATH = os.getenv("BIAS_TYPES_CSV_PATH")
EVALUATION_CSV_PATH = os.getenv("EVALUATION_CSV_PATH")
REFERENCES_CSV_PATH = os.getenv("REFERENCES_CSV_PATH")
REVIEWS_CSV_PATH = os.getenv("REVIEWS_CSV_PATH")
STUDIES_CSV_PATH = os.getenv("STUDIES_CSV_PATH")
CYPHER_CSV_PATH = os.getenv("CYPHER_CSV_PATH")

# Neo4j config
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Configure the logging module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


LOGGER = logging.getLogger(__name__)

NODES = ["SystematicReview", "Evaluation", "BiasType", "Study", "Reference"]

bias_types_df = pd.read_csv(BIAS_TYPES_CSV_PATH)
reviews_df = pd.read_csv(REVIEWS_CSV_PATH)
studies_df = pd.read_csv(STUDIES_CSV_PATH)
references_df = pd.read_csv(REFERENCES_CSV_PATH)
evaluations_df = pd.read_csv(EVALUATION_CSV_PATH)
cypher_df = pd.read_csv(CYPHER_CSV_PATH, sep=";")

def _set_uniqueness_constraints(tx, node):
    query = f"""CREATE CONSTRAINT IF NOT EXISTS FOR (n:{node})
        REQUIRE n.id IS UNIQUE;"""
    _ = tx.run(query, {})


@retry(tries=100, delay=10)
def load_cochrane_graph_from_csv() -> None:
    """Load structured Cochrane CSV data following
    a specific ontology into Neo4j"""

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    LOGGER.info("Setting uniqueness constraints on nodes")
    with driver.session(database="neo4j") as session:
        for node in NODES:
            session.execute_write(_set_uniqueness_constraints, node)

    LOGGER.info("Loading systematic review nodes")
    with driver.session(database="neo4j") as session:

        def bulk_insert_reviews(tx, reviews):
            query = f"""
            UNWIND $reviews AS review
            MERGE (:SystematicReview {{id: toInteger(review.index),
                                review_code: review.review_code,
                                title: review.title}});
            """
            tx.run(query, reviews=reviews)

        _ = session.execute_write(bulk_insert_reviews, reviews_df.to_dict(orient="records"))

    LOGGER.info("Loading bias types nodes")
    with driver.session(database="neo4j") as session:            
        def bulk_insert_bias_types(tx, bias_types):
            query = f"""
            UNWIND $bias_types AS bias_type
            MERGE (b:BiasType {{id: toInteger(bias_type.index)}})
            ON CREATE SET b.bias_id = bias_type.id,
                        b.name = bias_type.name,
                        b.description = bias_type.description,
                        b.group_id = bias_type.group_id,
                        b.group_description = bias_type.group_description
            ON MATCH SET  b.bias_id = bias_type.id,
                        b.name = bias_type.name,
                        b.description = bias_type.description,
                        b.group_id = bias_type.group_id,
                        b.group_description = bias_type.group_description
            """
            tx.run(query, bias_types=bias_types)
        _ = session.execute_write(bulk_insert_bias_types, bias_types_df.to_dict(orient="records"))

    LOGGER.info("Loading studies nodes")
    with driver.session(database="neo4j") as session:
        def bulk_insert_studies(tx, studies):
            query = f"""
            UNWIND $studies AS study
            MERGE (s:Study {{id: toInteger(study.index)}})
            ON CREATE SET s.review_index = study.review_index,
                        s.study_id = study.study_id,
                        s.study_name = study.study_name
            ON MATCH SET s.review_index = study.review_index,
                        s.study_id = study.study_id,
                        s.study_name = study.study_name
            """
            tx.run(query, studies=studies)
        _ = session.execute_write(bulk_insert_studies, studies_df.to_dict(orient="records"))
        

    LOGGER.info("Loading references nodes")
    with driver.session(database="neo4j") as session:
        def bulk_insert_references(tx, references):
            query = f"""
            UNWIND $references AS reference
            MERGE (r:Reference {{id: toInteger(reference.index)}})
            ON CREATE SET r.study_index = reference.study_index,        
                        r.title = reference.title,
                        r.authors = reference.authors,
                        r.journal = reference.journal,
                        r.pmcid = reference.pmcid,
                        r.pubmed = reference.pubmed
            ON MATCH SET r.study_index = reference.study_index,
                        r.title = reference.title,
                        r.authors = reference.authors,
                        r.journal = reference.journal,
                        r.pmcid = reference.pmcid,
                        r.pubmed = reference.pubmed
            """
            tx.run(query, references=references)
        _ = session.execute_write(bulk_insert_references, references_df.to_dict(orient="records"))

    LOGGER.info("Loading evaluation nodes")
    with driver.session(database="neo4j") as session:
        def bulk_insert_evaluations(tx, evaluations):
            query = f"""
            UNWIND $evaluations AS evaluation
            MERGE (e:Evaluation {{id: toInteger(evaluation.index)}})
            ON CREATE SET e.review_index = evaluation.review_index,
                                e.study_index = evaluation.study_index,
                                e.bias_id = evaluation.bias_id,
                                e.rob_judgment = evaluation.rob_judgment,
                                e.result = evaluation.result,
                                e.support_judgment = evaluation.support_judgment
            ON MATCH SET e.review_index = evaluation.review_index,
                                e.study_index = evaluation.study_index,
                                e.bias_id = evaluation.bias_id,
                                e.rob_judgment = evaluation.rob_judgment,
                                e.result = evaluation.result,
                                e.support_judgment = evaluation.support_judgment
            """
            tx.run(query, evaluations=evaluations)
        _ = session.execute_write(bulk_insert_evaluations, evaluations_df.to_dict(orient="records"))

    LOGGER.info("Loading question cypher nodes")
    with driver.session(database="neo4j") as session:
        def bulk_insert_cyphers(tx, cyphers):
            query = f"""
            UNWIND $cyphers AS cypher
            MERGE (c:Cypher {{question: cypher.question,
                              cypher: cypher.cypher}});
            """
            tx.run(query, cyphers=cyphers)

        _ = session.execute_write(bulk_insert_cyphers, cypher_df.to_dict(orient="records"))        


    LOGGER.info("Creating relationship SystematicReview CONTAINS Evaluation")
    with driver.session(database="neo4j") as session:
        def bulk_insert_contains_relationship(tx, evaluations):
            query = f"""
            UNWIND $evaluations AS evaluation
            MATCH (r:SystematicReview {{id: evaluation.review_index}})
            MATCH (e:Evaluation {{id: evaluation.index}})
            MERGE (r)-[:CONTAINS]->(e);
            """
            tx.run(query, evaluations=evaluations)

        _ = session.execute_write(bulk_insert_contains_relationship, evaluations_df.to_dict(orient="records"))

    LOGGER.info("Creating relationship Evaluation given BiasType")
    with driver.session(database="neo4j") as session:
        def bulk_insert_given_relationship(tx, evaluations):
            query = f"""
            UNWIND $evaluations AS evaluation
            MATCH (e:Evaluation {{id: evaluation.index}})
            MATCH (b:BiasType {{id: evaluation.bias_id}})
            MERGE (e)-[:GIVEN]->(b);
            """
            tx.run(query, evaluations=evaluations)

        _ = session.execute_write(bulk_insert_given_relationship, evaluations_df.to_dict(orient="records"))

    LOGGER.info("Creating relationship Evaluation on Study")
    with driver.session(database="neo4j") as session:
        def bulk_insert_on_relationship(tx, evaluations):
            query = f"""
            UNWIND $evaluations AS evaluation
            MATCH (e:Evaluation {{id: evaluation.index}})
            MATCH (s:Study {{id: evaluation.study_index}})
            MERGE (e)-[:ON]->(s);
            """
            tx.run(query, evaluations=evaluations)

        _ = session.execute_write(bulk_insert_on_relationship, evaluations_df.to_dict(orient="records"))

    LOGGER.info("Creating relationship Study cites Reference")
    with driver.session(database="neo4j") as session:
        def bulk_insert_cites_relationship(tx, references):
            query = f"""
            UNWIND $references AS reference
            MATCH (s:Study {{id: reference.study_index}})
            MATCH (r:Reference {{id: reference.index}})
            MERGE (s)-[:CITES]->(r);
            """
            tx.run(query, references=references)

        _ = session.execute_write(bulk_insert_cites_relationship, references_df.to_dict(orient="records"))

    LOGGER.info("Creating relationship Review includes Study")
    with driver.session(database="neo4j") as session:
        def bulk_insert_includes_relationship(tx, studies):
            query = f"""
            UNWIND $studies AS study
            MATCH (r:SystematicReview {{id: study.review_index}})
            MATCH (s:Study {{id: study.index}})
            MERGE (r)-[:INCLUDES]->(s);
            """
            tx.run(query, studies=studies)

        _ = session.execute_write(bulk_insert_includes_relationship, studies_df.to_dict(orient="records"))

if __name__ == "__main__":
    load_cochrane_graph_from_csv()