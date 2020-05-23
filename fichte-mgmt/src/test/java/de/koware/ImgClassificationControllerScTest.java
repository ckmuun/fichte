package de.koware;

import io.quarkus.test.junit.QuarkusTest;
import org.junit.jupiter.api.Test;

import static io.restassured.RestAssured.given;
import static org.hamcrest.CoreMatchers.is;


@QuarkusTest
class ImgClassificationControllerScTest {


    @Test
    public void testHelloEndpoint() {
        given()
                .when().get("/classification-dashboard")
                .then()
                .statusCode(200)
                .body(is("hello classification"));
    }


}