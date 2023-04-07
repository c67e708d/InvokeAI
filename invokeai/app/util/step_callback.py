# import torch
from invokeai.app.datatypes.exceptions import CanceledException

from invokeai.app.services.events import ProgressImage
from ..invocations.baseinvocation import InvocationContext
from ...backend.util.util import image_to_dataURL
from ...backend.generator.base import Generator
from ...backend.stable_diffusion import PipelineIntermediateState


def step_callback(
    context: InvocationContext,
    intermediate_state: PipelineIntermediateState,
    total_steps: int,
    node_id: str,
) -> None:
    if context.services.queue.is_canceled(context.graph_execution_state_id):
        raise CanceledException

    step = intermediate_state.step

    if intermediate_state.predicted_original is not None:
        # Some schedulers report not only the noisy latents at the current timestep,
        # but also their estimate so far of what the de-noised latents will be.
        sample = intermediate_state.predicted_original
    else:
        sample = intermediate_state.latents

    # TODO: only output a preview image when requested
    image = Generator.sample_to_lowres_estimated_image(sample)

    progress_image: ProgressImage = {
        "width": image.size[0] * 8,
        "height": image.size[1] * 8,
        "dataURL": image_to_dataURL(image, image_format="JPEG"),
    }

    graph_execution_state = context.services.graph_execution_manager.get(
        context.graph_execution_state_id
    )

    invocation = graph_execution_state.execution_graph.get_node(node_id)

    source_id = graph_execution_state.prepared_source_mapping.get(invocation.id, None)

    context.services.events.emit_generator_progress(
        graph_execution_state_id=context.graph_execution_state_id,
        invocation=invocation.dict(),
        source_id=source_id,
        progress_image=progress_image,
        step=step,
        total_steps=total_steps,
    )
