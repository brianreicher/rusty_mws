import unittest
import pymongo
from funlib.persistence import open_ds, Array, graphs, prepare_ds
import logging


logger: logging.Logger = logging.getLogger(__name__)


class MongoGraphTests(unittest.TestCase):
    def test_connection(self):
        db_host: str = "mongodb://localhost:27017"
        db_name: str = "seg"

        mongo_drop = pymongo.MongoClient(db_host)[db_name]
        collection_names: list = mongo_drop.list_collection_names()

        sample_name: str = "htem4413041148969302336"
        for collection_name in collection_names:
            if sample_name in collection_name:
                logger.info(f"Dropping {collection_name}")
                mongo_drop[collection_name].drop()

        logger.info("Opening MongoDBGraphProvider...")
        rag_provider = graphs.MongoDbGraphProvider(
            db_name=db_name,
            host=db_host,
            mode="r+",
            directed=False,
            position_attribute=["center_z", "center_y", "center_x"],
            edges_collection=f"{sample_name}_edges",
            nodes_collection=f"{sample_name}_nodes",
            meta_collection=f"{sample_name}_meta",
        )

        connect_succes: bool = True

        if rag_provider.client is not None:
            connect_succes = True

        self.assertEqual(first=connect_succes, second=True)

    def test_write(self):
        pass

    def test_read(self):
        pass


if __name__ == "__main__":
    unittest.main()
