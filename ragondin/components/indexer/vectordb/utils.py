from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
    relationship,
)


from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    ForeignKey,
    DateTime,
    UniqueConstraint,
    func,
)

from datetime import datetime
from pydantic import BaseModel
from typing import List
from loguru import logger


Base = declarative_base()


class FileModel(BaseModel):
    id: int
    file_id: str
    partition: str


class BasePartitionModel(BaseModel):
    partition: str
    created_at: datetime


# In the PartitionModel class
class PartitionModel(BasePartitionModel):
    files: List[FileModel] = []


class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True)
    file_id = Column(String, nullable=False)
    # Foreign key points directly to the partition string
    partition_name = Column(String, ForeignKey("partitions.partition"), nullable=False)

    # relationship to the Partition object
    partition = relationship("Partition", back_populates="files")

    # Enforce uniqueness of (file_id, partition_name)
    __table_args__ = (
        UniqueConstraint("file_id", "partition_name", name="uix_file_id_partition"),
    )

    def to_pydantic(self):
        return FileModel(id=self.id, file_id=self.file_id, partition=self.partition)

    def __repr__(self):
        return f"<File(id={self.id}, file_id='{self.file_id}', partition='{self.partition}')>"


# In the Partition model
class Partition(Base):
    __tablename__ = "partitions"

    id = Column(Integer, primary_key=True)
    partition = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    files = relationship(
        "File", back_populates="partition", cascade="all, delete-orphan"
    )

    def to_pydantic(self, exclude_files=True):
        if exclude_files:
            return BasePartitionModel(
                partition=self.partition, created_at=self.created_at
            )
        else:
            return PartitionModel(
                partition=self.partition,
                created_at=self.created_at,
                files=[file.to_pydantic() for file in self.files],
            )

    def __repr__(self):
        return f"<Partition(key='{self.partition}', created_at='{self.created_at}', file_count={len(self.files)})>"


class PartitionFileManager:
    def __init__(self, database_url: str, logger=logger):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.logger = logger

    def get_partition(self, partition: str):
        """Retrieve a partition by its key"""
        with self.Session() as session:
            self.logger.debug(f"Fetching partition with key: {partition}")
            partition = session.query(Partition).filter_by(partition=partition).first()
            if partition:
                self.logger.info(f"Partition found: {partition}")
            else:
                self.logger.warning(f"No partition found with key: {partition}")
            return partition

    # def create_partition(self, partition: str):
    #     """Create a new partition if it doesn't exist"""
    #     with self.Session() as session:
    #         try:
    #             partition = (
    #                 session.query(Partition).filter_by(partition=partition).first()
    #             )
    #             if not partition:
    #                 partition = Partition(partition=partition)
    #                 session.add(partition)
    #                 session.commit()
    #                 self.logger.info(f"Created new partition with key: {partition}")
    #             return partition
    #         except Exception as e:
    #             session.rollback()
    #             raise e

    def add_file_to_partition(self, file_id: str, partition: str):
        """Add a file to a partition"""
        with self.Session() as session:
            try:
                # Check if file already exists in this partition
                existing_file = (
                    session.query(File)
                    .join(Partition)
                    .filter(File.file_id == file_id, Partition.partition == partition)
                    .first()
                )
                if existing_file:
                    self.logger.warning(
                        f"File {file_id} already exists in partition {partition}"
                    )
                    return False

                # Ensure partition exists
                partition_obj = (
                    session.query(Partition).filter_by(partition=partition).first()
                )
                if not partition_obj:
                    partition_obj = Partition(partition=partition)
                    session.add(partition_obj)
                    self.logger.info(f"Created new partition with key: {partition}")

                # Add file to partition
                file = File(file_id=file_id, partition_name=partition_obj.partition)
                session.add(file)
                session.commit()
                self.logger.info(f"Added file {file_id} to partition {partition}")
                return True
            except Exception as e:
                session.rollback()
                self.logger.error(f"Error adding file to partition: {e}")
                raise e

    def remove_file_from_partition(self, file_id: str, partition: str):
        """Remove a file from its partition"""
        with self.Session() as session:
            try:
                # Find the file using a join with proper filtering
                file = (
                    session.query(File)
                    .join(Partition)
                    .filter(File.file_id == file_id, Partition.partition == partition)
                    .first()
                )
                if file:
                    session.delete(file)
                    session.commit()
                    self.logger.info(
                        f"Removed file {file_id} from partition {partition}"
                    )

                    # Check if partition is now empty
                    partition_obj = (
                        session.query(Partition).filter_by(partition=partition).first()
                    )
                    if partition_obj and len(partition_obj.files) == 0:
                        session.delete(partition_obj)
                        session.commit()
                        self.logger.info(f"Deleted empty partition {partition}")

                    return True
                self.logger.warning(
                    f"File {file_id} not found in partition {partition}"
                )
                return False
            except Exception as e:
                session.rollback()
                self.logger.error(f"Error removing file: {e}")
                raise e

    def delete_partition(self, partition: str):
        """Delete a partition and all its files"""
        with self.Session() as session:
            partition = session.query(Partition).filter_by(partition=partition).first()
            if partition:
                session.delete(partition)  # Will delete all files due to cascade
                session.commit()
                self.logger.info(f"Deleted partition with key: {partition}")
                return True
            else:
                self.logger.info(f"There is no partition named `{partition}`")
            return False

    def list_partitions(self):
        """List all existing partitions"""
        with self.Session() as session:
            partitions = session.query(Partition).all()
            return [
                partition.to_pydantic(exclude_files=True) for partition in partitions
            ]

    def list_files_in_partition(self, partition: str):
        """List all files in a partition"""
        with self.Session() as session:
            partition = session.query(Partition).filter_by(partition=partition).first()
            if partition:
                return [file.file_id for file in partition.files]
            return []

    def get_partition_file_count(self, partition: str):
        """Get the count of files in a partition"""
        with self.Session() as session:
            partition_obj = (
                session.query(Partition).filter_by(partition=partition).first()
            )
            if not partition_obj:
                return 0
            return len(partition_obj.files)  # Or use a count query if you prefer

    def get_total_file_count(self):
        """Get the total count of files across all partitions"""
        with self.Session() as session:
            return session.query(File).count()

    def partition_exists(self, partition: str):
        """Check if a partition exists by its key"""
        with self.Session() as session:
            return session.query(Partition).filter_by(partition=partition).count() > 0

    def file_exists_in_partition(self, file_id: str, partition: str):
        """Check if a file exists in a specific partition"""
        with self.Session() as session:
            # Use a join to correctly filter by both file_id and partition string
            return (
                session.query(File)
                .join(Partition)  # Join the File and Partition tables
                .filter(
                    File.file_id == file_id,
                    Partition.partition == partition,  # Filter on the partition string
                )
                .count()
                > 0
            )

    def sample_file_ids(self, partition: str, n_file_id: int = 1000, seed: int = None):
        """Sample file IDs from a partition"""
        with self.Session() as session:
            # Single query to directly get file_ids from the specified partition
            random_func = self.get_random_func(session, seed)
            sampled_files = (
                session.query(File.file_id)
                .filter(File.partition_name == partition)
                .order_by(random_func)
                .limit(n_file_id)
                .all()
            )

            if not sampled_files:
                self.logger.warning(f"No files found in partition: {partition}")
                return []

            # Extract file_ids from result tuples
            return [file.file_id for file in sampled_files]

    def get_random_func(self, session, seed: int = None):
        if seed is not None:
            db_type = self.engine.name
            match db_type:
                case "postgresql":
                    # PostgreSQL: First set the seed, then use random()
                    session.execute(f"SELECT setseed({seed / ((2**31) - 1)})")
                    return func.random()
                case "'mysql":
                    # MySQL: Use RAND() with a seed
                    return func.rand(seed)
                case "sqlite":
                    # SQLite: Use random() with a seed
                    return func.random(seed)
                case _:
                    # Other databases: Use standard random with warning
                    self.logger.warning(
                        f"Database type '{db_type}' not supported for seeded random sampling."
                    )
                    return func.random()
        else:
            return func.random()
