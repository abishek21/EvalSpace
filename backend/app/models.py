from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from enum import Enum


class JobStatus(str, Enum):
    queued = "queued"
    downloading = "downloading"
    generating = "generating"
    completed = "completed"
    failed = "failed"


class DatasetColumnMap(BaseModel):
    """Maps generic column roles to actual dataset column names."""
    image_column: str = "image"
    text_column: str = "text"
    scene_id_column: Optional[str] = None  # If dataset has a scene/group ID


class JobConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    dataset: str = "ZiAngGu/scannet_3dbox_v2"
    split: str = "train"
    num_scenes: int = Field(5, alias="numScenes")
    questions_per_scene: int = Field(10, alias="questionsPerScene")
    categories: list[str] = ["counting", "spatial", "occlusion", "affordance", "manipulation", "scene"]
    model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    max_views: int = Field(3, alias="maxViews")
    image_resolution: int = Field(480, alias="imageResolution")
    name: str | None = Field(None)
    job_type: str | None = Field(None, alias="jobType")
    dataset_id: str | None = Field(None, alias="datasetId")
    preferred_run_id: str | None = Field(None, alias="preferredRunId")
    rejected_run_id: str | None = Field(None, alias="rejectedRunId")
    strategy: str | None = Field(None)
    objects_per_scene: int = Field(4, alias="objectsPerScene")
    use_curated: bool = Field(False, alias="useCurated")
    num_stable: int | None = Field(None, alias="numStable")
    num_unstable: int | None = Field(None, alias="numUnstable")
    environment: str | None = Field(None)
    question_model: dict | None = Field(None, alias="questionModel")
    azure_config: dict | None = Field(None, alias="azureConfig")
    column_map: DatasetColumnMap = Field(default_factory=DatasetColumnMap, alias="columnMap")

    # Customizable system prompts
    question_system_prompt: str = Field(
        "Generate spatial reasoning questions about 3D scenes for robotics RLHF. Output ONLY valid JSON.",
        alias="questionSystemPrompt",
    )
    answer_system_prompt: str = Field(
        "You are a spatial reasoning VLM for robotics. You analyze 3D scenes from multiple camera views "
        "and answer questions with precise step-by-step reasoning.\n\n"
        "Always follow Chain-of-Thought:\n"
        "1. **Observe**: Describe what you see in the images relevant to the question\n"
        "2. **Reason**: Apply spatial logic — consider what's visible, occluded, reachable\n"
        "3. **Answer**: Give a clear, precise final answer\n\n"
        "Be accurate. If something is partially occluded or uncertain from the views, say so.",
        alias="answerSystemPrompt",
    )
    rejected_system_prompt: str = Field(
        "Answer questions about rooms briefly and directly. Don't overthink it.",
        alias="rejectedSystemPrompt",
    )
    question_user_prompt_template: str = Field(
        "Scene: {descriptions}\n\n"
        "Generate {n} questions across these categories: {categories}.\n"
        "Vary difficulty: easy, medium, hard.\n\n"
        'Output JSON:\n{{"questions": [{{"id": 1, "text": "...", "category": "...", "difficulty": "..."}}]}}',
        alias="questionUserPromptTemplate",
    )
    chosen_user_prompt_template: str = Field(
        "Scene annotations: {descriptions}\n\n"
        "Question: {question}\n\n"
        "Look at the provided camera views carefully and respond with step-by-step spatial reasoning (Observe → Reason → Answer).",
        alias="chosenUserPromptTemplate",
    )
    rejected_user_prompt_template: str = Field(
        "Room with objects: {descriptions_brief}\nQuestion: {question}\nAnswer:",
        alias="rejectedUserPromptTemplate",
    )


class JobProgress(BaseModel):
    scenes_processed: int = 0
    questions_generated: int = 0
    pairs_generated: int = 0


class Job(BaseModel):
    id: str
    status: JobStatus = JobStatus.queued
    config: JobConfig
    progress: JobProgress = Field(default_factory=JobProgress)
    project_id: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class PairSource(BaseModel):
    dataset: str
    split: str
    row_indices: list[int]
    scene_id: str
    images: list[str]  # base64 data URIs


class PairGeneration(BaseModel):
    model: str
    chosen_strategy: str
    rejected_strategy: str
    chosen_temperature: float
    rejected_temperature: float
    num_views: int
    image_resolution: int
    generated_at: str


class Pair(BaseModel):
    id: str
    project_id: str
    prompt: str
    chosen: str
    rejected: str
    scene_id: str
    category: str
    difficulty: str
    status: str = "pending"  # pending | annotated | skipped
    preference: Optional[str] = None  # chosen | rejected | tie
    rationale: Optional[str] = None
    annotated_at: Optional[str] = None
    source: Optional[PairSource] = None
    generation: Optional[PairGeneration] = None


class Project(BaseModel):
    id: str
    name: str
    created_at: str
    job_id: Optional[str] = None
