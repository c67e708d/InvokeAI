# Invocations

WIP

## Schema Generation

Invocation, output and related classes are used to generate an OpenAPI schema.

### Required Properties

The schema generation treat all properties with default values as optional. This
makes sense internally, but when when using these classes via the generated
schema, we end up with e.g. the `ImageOutput` class having its `image` property
marked as optional.

We know that this property will always be present, so the additional logic
needed to always check if the property exists adds a lot of extraneous cruft.

To fix this, we can leverage `pydantic`'s
[schema customisation](https://docs.pydantic.dev/usage/schema/#schema-customization)
to mark properties that we know will always be present as required.

Here's that `ImageOutput` class, without the needed schema customisation:

```python
class ImageOutput(BaseInvocationOutput):
    """Base class for invocations that output an image"""

    type: Literal["image"] = "image"
    image:      ImageField = Field(default=None, description="The output image")
```

The generated OpenAPI schema, and all clients/types generated from it, will have
the `type` and `image` properties marked as optional, even though we know they
will always have a value by the time we can interact with them via the API.

Here's the same class, but with the schema customisation added:

```python
class ImageOutput(BaseInvocationOutput):
    """Base class for invocations that output an image"""

    type: Literal["image"] = "image"
    image:      ImageField = Field(default=None, description="The output image")

    class Config:
        schema_extra = {
            'required': [
                'type',
                'image',
            ]
        }
```

The resultant schema (and any API client or types generated from it) will now
have see `type` as string literal `"image"` and `image` as an `ImageField`
object.

See this `pydantic` issue for discussion on this solution:
<https://github.com/pydantic/pydantic/discussions/4577>
