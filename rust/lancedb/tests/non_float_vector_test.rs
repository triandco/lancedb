use std::sync::Arc;

use arrow::datatypes::{Float32Type, Int8Type};
use arrow_array::{Array, FixedSizeListArray, Int32Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::{arrow::IntoArrow, connect, error::Result, query::{ExecutableQuery, QueryBase}};
use tempfile::TempDir;

fn create_records(data_type: DataType, vectors: FixedSizeListArray, vector_dimension: i32) -> Result<impl IntoArrow> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", data_type, true)),
                vector_dimension,
            ),
            true,
        ),
    ]));

    // Create a RecordBatch stream.
    let batches = RecordBatchIterator::new(
        vec![RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..vectors.len() as i32)),
                Arc::new(vectors),
            ],
        )
        .unwrap()]
        .into_iter()
        .map(Ok),
        schema.clone(),
    );
    Ok(Box::new(batches))
}

fn create_float32_vector(count: usize, vector_dimension: usize) -> FixedSizeListArray {
    FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        (0..count).map(|_| Some(vec![Some(1f32); vector_dimension])),
        vector_dimension as i32,
    )
}

fn create_int8_vector(count: usize, vector_dimension: usize) -> FixedSizeListArray {
    FixedSizeListArray::from_iter_primitive::<Int8Type, _, _>(
        (0..count).map(|_| Some(vec![Some(1i8); vector_dimension])),
        vector_dimension as i32,
    )
}

#[tokio::test]
async fn index_floating_type_no_index() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let db = connect(temp_dir.path().to_str().unwrap()).execute().await?;
    let initial_data = create_records(DataType::Float32, create_float32_vector(1000, 128), 128)?;
    let table = db
        .create_table("my_table", initial_data)
        .execute()
        .await
        .unwrap();

    let record_batch = table
        .query()
        .limit(2)
        .nearest_to(&[1f32; 128])?
        .column("vector")
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    
    let expected = record_batch
      .column_by_name("id")
      .unwrap()
      .as_any()
      .downcast_ref::<Int32Array>()
      .unwrap()
      .into_iter()
      .next()
      .unwrap()
      .to_owned()
      .unwrap();

    assert!(expected == 0, "Expect found record id 0 got {expected}");
    Ok(())
}

#[tokio::test]
async fn index_floating_type_with_index() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let db = connect(temp_dir.path().to_str().unwrap()).execute().await?;
    let initial_data = create_records(DataType::Float32, create_float32_vector(1000, 128), 128)?;
    let table = db
        .create_table("my_table", initial_data)
        .execute()
        .await
        .unwrap();

    table.create_index(&["vector"], lancedb::index::Index::Auto).execute().await?;

    let record_batch = table
        .query()
        .limit(2)
        .nearest_to(&[1f32; 128])?
        .column("vector")
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    
    let expected = record_batch
      .column_by_name("id")
      .unwrap()
      .as_any()
      .downcast_ref::<Int32Array>()
      .unwrap()
      .into_iter()
      .next()
      .unwrap()
      .to_owned()
      .unwrap();

    assert!(expected == 65, "Expect found record id 0 got {expected}");
    Ok(())
}

#[tokio::test]
async fn index_non_floating_type_no_index() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let db = connect(temp_dir.path().to_str().unwrap()).execute().await?;
    let initial_data = create_records(DataType::Int8, create_int8_vector(1000, 128), 128)?;
    let table = db
        .create_table("my_table", initial_data)
        .execute()
        .await
        .unwrap();

    let record_batch = table
        .query()
        .limit(2)
        .nearest_to(&[1f32; 128])?
        .column("vector")
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    
    let expected = record_batch
      .column_by_name("id")
      .unwrap()
      .as_any()
      .downcast_ref::<Int32Array>()
      .unwrap()
      .into_iter()
      .next()
      .unwrap()
      .to_owned()
      .unwrap();

    assert!(expected == 0, "Expect found record id 0 got {expected}");
    Ok(())
}

#[tokio::test]
async fn index_non_floating_type_with_index() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let db = connect(temp_dir.path().to_str().unwrap()).execute().await?;
    let initial_data = create_records(DataType::Int8, create_int8_vector(1000, 128), 128)?;
    let table = db
        .create_table("my_table", initial_data)
        .execute()
        .await
        .unwrap();

    table.create_index(&["vector"], lancedb::index::Index::Auto).execute().await?;

    let record_batch = table
        .query()
        .limit(2)
        .nearest_to(&[1f32; 128])?
        .column("vector")
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    
    let expected = record_batch
      .column_by_name("id")
      .unwrap()
      .as_any()
      .downcast_ref::<Int32Array>()
      .unwrap()
      .into_iter()
      .next()
      .unwrap()
      .to_owned()
      .unwrap();

    assert!(expected == 533, "Expect found record id 0 got {expected}");
    Ok(())
}
