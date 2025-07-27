package org.example;

import io.milvus.client.MilvusClient;
import io.milvus.client.MilvusServiceClient;
import io.milvus.grpc.*;
import io.milvus.param.*;
import io.milvus.param.collection.*;
import io.milvus.param.dml.InsertParam;
import io.milvus.param.dml.SearchParam;
import io.milvus.param.index.CreateIndexParam;
import io.milvus.response.SearchResultsWrapper;

import java.util.Arrays;
import java.util.List;

public class Main {

    private static final String MY_COLLECTION = "my_collection";
    private static MilvusClient milvusClient;

    public static void main(String[] args) {
        connectToMilvus();
        createCollectionIfNotExists();
        insertVectorsIfEmpty();
        createIndexIfNotExists();
        searchSimilarVectors(Arrays.asList(0.1f, 0.2f, 0.3f, 0.4f), 5);
    }

    private static void connectToMilvus() {
        milvusClient = new MilvusServiceClient(ConnectParam.newBuilder()
                .withHost("localhost")
                .withPort(19530)
                .withAuthorization("root", "Milvus")
                .build());

        System.out.println("Connection established with Milvus.");
    }

    private static void createCollectionIfNotExists() {
        HasCollectionParam hasCollectionParam = HasCollectionParam.newBuilder()
                .withCollectionName(MY_COLLECTION)
                .build();

        boolean exists = milvusClient.hasCollection(hasCollectionParam).getData();

        if (!exists) {
            System.out.println("Collection not found. Creating...");

            FieldType idField = FieldType.newBuilder()
                    .withName("id")
                    .withDataType(DataType.Int64)
                    .withPrimaryKey(true)
                    .withAutoID(false)
                    .build();

            FieldType vectorField = FieldType.newBuilder()
                    .withName("embedding")
                    .withDescription("Vector field")
                    .withDataType(DataType.FloatVector)
                    .withDimension(4)
                    .withPrimaryKey(false)
                    .withAutoID(false)
                    .build();

            CreateCollectionParam createCollectionParam = CreateCollectionParam.newBuilder()
                    .withCollectionName(MY_COLLECTION)
                    .withDescription("A sample vector collection")
                    .withShardsNum(2)
                    .addFieldType(idField)
                    .addFieldType(vectorField)
                    .build();

            milvusClient.createCollection(createCollectionParam);
            System.out.println("Collection created successfully.");
        } else {
            System.out.println("Collection already exists. Skipping creation.");
        }
    }

    private static void insertVectorsIfEmpty() {
        if (!isCollectionEmpty()) {
            System.out.println("Collection already contains data. Skipping insert.");
            return;
        }

        System.out.println("Inserting sample vectors...");

        List<Long> ids = Arrays.asList(1L, 2L, 3L);

        List<List<Float>> vectors = Arrays.asList(
                Arrays.asList(0.1f, 0.2f, 0.3f, 0.4f),
                Arrays.asList(0.5f, 0.6f, 0.7f, 0.8f),
                Arrays.asList(0.9f, 1.0f, 1.1f, 1.2f)
        );

        InsertParam.Field idField = InsertParam.Field.builder()
                .name("id")
                .values(ids)
                .build();

        InsertParam.Field vectorField = InsertParam.Field.builder()
                .name("embedding")
                .values(vectors)
                .build();

        InsertParam insertParam = InsertParam.newBuilder()
                .withCollectionName(MY_COLLECTION)
                .withFields(Arrays.asList(idField, vectorField))
                .build();

        R<MutationResult> response = milvusClient.insert(insertParam);

        if (response.getStatus() != R.Status.Success.getCode()) {
            System.err.println("Insertion failed: " + response.getException().getMessage());
        } else {
            System.out.println("Entered " + response.getData().getInsertCnt() + " vettori.");
        }
    }

    private static boolean isCollectionEmpty() {
        R<GetCollectionStatisticsResponse> stats = milvusClient.getCollectionStatistics(
                GetCollectionStatisticsParam.newBuilder()
                        .withCollectionName(Main.MY_COLLECTION)
                        .build()
        );

        if (stats.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException("Unable to get collection statistics: " + stats.getException().getMessage());
        }

        return stats.getData().getStatsList().stream()
                .filter(pair -> pair.getKey().equals("row_count"))
                .map(KeyValuePair::getValue)
                .findFirst()
                .map(count -> Long.parseLong(count) == 0)
                .orElse(true);
    }

    private static void createIndexIfNotExists() {
        CreateIndexParam createIndexParam = CreateIndexParam.newBuilder()
                .withCollectionName(MY_COLLECTION)
                .withFieldName("embedding")
                .withIndexName("embedding_index")
                .withIndexType(IndexType.IVF_FLAT)
                .withMetricType(MetricType.L2)
                .withExtraParam("{\"nlist\":128}")
                .withSyncMode(true)
                .build();

        R<RpcStatus> response = milvusClient.createIndex(createIndexParam);

        if (response.getStatus() != R.Status.Success.getCode()) {
            System.err.println("Index creation failed: " + response.getException().getMessage());
        } else {
            System.out.println("Index created successfully.");
        }
    }


    public static void searchSimilarVectors(List<Float> queryVector, int topK) {
        milvusClient.loadCollection(
                LoadCollectionParam.newBuilder()
                        .withCollectionName(MY_COLLECTION)
                        .build()
        );

        SearchParam searchParam = SearchParam.newBuilder()
                .withCollectionName(MY_COLLECTION)
                .withMetricType(MetricType.L2)
                .withOutFields(List.of("id"))
                .withTopK(topK)
                .withVectors(List.of(queryVector))  // deve essere una lista di liste
                .withVectorFieldName("embedding")
                .build();

        R<SearchResults> response = milvusClient.search(searchParam);

        if (response.getStatus() != R.Status.Success.getCode()) {
            System.err.println("Search failed: " + response.getException().getMessage());
            return;
        }

        SearchResultsWrapper wrapper = new SearchResultsWrapper(response.getData().getResults());
        List<SearchResultsWrapper.IDScore> scores = wrapper.getIDScore(0);

        System.out.println("Search results: ");
        for (SearchResultsWrapper.IDScore result : scores) {
            System.out.printf("ID: %d, Distance: %.4f%n", result.getLongID(), result.getScore());
        }
    }
}
