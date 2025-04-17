from sqlalchemy import create_engine, Column, Integer, String, func
from sqlalchemy.orm import declarative_base, sessionmaker, Session as SessionType

from pydantic import BaseModel
from loguru import logger


Base = declarative_base()


class PartitionModel(BaseModel):
    partition_key: str
    count: int = 0


class Partition(Base):
    __tablename__ = "partitions"

    id = Column(Integer, primary_key=True)
    partition_key = Column(String, unique=True, nullable=False)
    count = Column(Integer, default=1)

    def to_pydantic(self):
        return PartitionModel(partition_key=self.partition_key, count=self.count)

    def __repr__(self):
        return f"<Partition(key='{self.partition_key}', count={self.count}>"


class PartitionKeyManager:
    def __init__(self, database_url: str, logger=logger):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.logger = logger

    def get_partition(self, partition_key: str):
        """Retrieve a partition by its key"""
        # partition_key = partition_key.lower()
        with self.Session() as session:
            self.logger.debug(f"Fetching partition with key: {partition_key}")
            partition = (
                session.query(Partition).filter_by(partition_key=partition_key).first()
            )
            if partition:
                self.logger.info(f"Partition found: {partition}")
            else:
                self.logger.warning(f"No partition found with key: {partition_key}")
            return partition

    def create_or_update_partition(self, partition_key: str, amount: int = 1):
        """Increment the count for a partition"""
        # partition_key = partition_key.lower()
        with self.Session() as session:
            try:
                partition = (
                    session.query(Partition)
                    .filter_by(partition_key=partition_key)
                    .first()
                )
                if partition:
                    partition.count += amount
                    self.logger.debug(
                        f"Updated partition count to {partition.count} for key: {partition_key}"
                    )

                    if partition.count <= 0:
                        session.delete(partition)
                        self.logger.info(
                            f"Deleted partition with key: {partition_key} due to non-positive count."
                        )
                else:
                    partition = Partition(partition_key=partition_key, count=amount)
                    session.add(partition)
                    self.logger.info(f"Created new partition with key: {partition_key}")

                session.commit()
            except Exception as e:
                session.rollback()
                raise e

    def delete_partition(self, partition_key: str):
        """Delete a partition by its key"""
        # partition_key = partition_key.lower()
        with self.Session() as session:
            partition = (
                session.query(Partition).filter_by(partition_key=partition_key).first()
            )
            if partition:
                session.delete(partition)
                session.commit()
                self.logger.info(f"Deleted partition with key: {partition_key}")
                return True

            return False

    def list_partitions(self):
        """List all existing partitions"""
        with self.Session() as session:
            # convert to Pydantic models
            partitions = session.query(Partition).all()
            return [partition.to_pydantic().partition_key for partition in partitions]

    def partition_exists(self, partition_key: str):
        """Check if a partition exists by its key"""
        with self.Session() as session:
            return (
                session.query(Partition).filter_by(partition_key=partition_key).count()
                > 0
            )

    def get_total_count(self):
        """Get the total count across all partitions"""
        with self.Session() as session:
            return (
                session.query(Partition)
                .with_entities(func.sum(Partition.count))
                .scalar()
                or 0
            )
