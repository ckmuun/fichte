package modelServe;

import javax.annotation.ParametersAreNonnullByDefault;
import javax.annotation.Resource;
import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;

@Path("/classification-dashboard")
public class ImgClassificationController {


    @GET
    @Produces(MediaType.TEXT_PLAIN)
    public String getHello() {
        return "hello classification";
    }
}
