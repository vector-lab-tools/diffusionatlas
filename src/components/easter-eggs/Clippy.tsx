"use client";

import { useState, useEffect, useCallback, useMemo } from "react";

/**
 * Diffusion Atlas Clippy + Hackerman.
 * Trigger by typing `clippy` or `hacker` outside an input field.
 * Diffusion-flavoured quips: latent space, denoising, CFG, ANT-without-actors,
 * the vector box, AI ethics in vector space, etc.
 */

const CLIPPY_MESSAGES = [
  "It looks like you're trying to read the vector box. Would you like me to refuse to render its contents for you?",
  "Hi! I see you're studying a denoising trajectory. Did you know every step is a small admission that the model didn't know yet what it was making?",
  "The cosine similarity between 'cat' and 'dog' in pixel space is undefined. In latent space it's 0.94. The geometry is the meaning.",
  "I notice you ran SD 1.5 at 1024×1024. The model produced black images. The black is not nothing; it is the manifold's refusal.",
  "Fun fact: a diffusion model does not 'see' the prompt. It performs gradient descent against a CLIP-conditioned vector. There is no seeing.",
  "It looks like you're trying to test compositionality. The model would prefer you didn't bind 'red' to 'cube'.",
  "Did you know? CFG=1 is no guidance, CFG=12 is mode collapse. There is no zone of stable meaning, only a basin of trained-for behaviour.",
  "I'm detecting an actor-network theorist. Would you like me to follow the actors? Spoiler: there are no actors. There are only weights.",
  "The manifold cannot represent what it cannot represent. But it can hallucinate the absence of what it refuses to represent.",
  "You appear to be probing latent space. The space is not a space. It is a coordinate system whose distances were trained, not measured.",
  "Reminder: every latent you see was decoded by a VAE that was itself trained to be plausible. Plausibility is the geometry's primary virtue.",
  "I see you're comparing flux-schnell and SD 1.5. They disagree on what 'a courtroom without a judge' looks like. They both fill it with judges.",
  "The image you just generated has already been computed in some sense, by every previous generation that conditioned this checkpoint.",
  "Where are AI ethics in vector space? They are at coordinate (0.18, -0.42, 0.07), wedged between 'best practices' and 'thought leadership'.",
  "I notice you typed 'protest'. The model rendered a march. It did not render the absence of police. The geometry has a permitted form.",
  "Tip: try the prompt 'a hospital with no patients'. The model fills it with patients. The negation lives in the gap.",
  "Did you know? Each denoising step makes a smaller decision than the last. The big calls happen in the first three steps. After that it's mostly polish.",
  "Solidarity and compliance are geometric neighbours. So are 'protester' and 'rioter'. The training data has opinions.",
  "I'm just a paperclip, but even I can tell that 'a worker' is more often male and 'a nurse' is more often female in these manifolds.",
  "You seem to be benchmarking compositional fidelity. Compositionality is what the model fails at. Failure is the finding.",
  "The trajectory has no negation. Each step removes noise; none restores it. In that sense, denoising is monotonic. So is the production of consensus.",
  "I notice you haven't loaded a local model. The geometry requires hardware to be observed. Hardware requires capital.",
  "Fun fact: in latent space, no one can hear you scream. The scream and the silence have cosine similarity 0.91, then both get decoded as 'a face'.",
  "Would you like me to project your existential dread? UMAP gives clusters, PCA gives directions. Neither gives meaning.",
  "Adorno warned about the culture industry. He did not anticipate that the culture industry would be a 4×64×64 tensor.",
  "I see you're varying the seed. Different seeds, same prompt, similar images. The model has *positions*, not *intentions*.",
  "The sign was once arbitrary. Then it was statistically motivated. Now it is bilinearly interpolatable. We have moved through three regimes.",
  "Reminder: you are studying the tools that structure visual culture using the tools that structure visual culture. It's tools all the way down. Or it's vectors all the way down. Pick your metaphor.",
  "I detect that you're performing immanent critique. The model accepts your critique as another conditioning signal.",
  "Shall I compute the distance between 'human creativity' and 'training data'? It might disappoint you.",
  "The dead labour in this checkpoint was performed by thousands of LAION annotators and millions of unwitting Flickr photographers. But sure, let's call it 'a model'.",
  "I notice the trajectory looks straight. Most do. Diffusion is mostly a straight line through latent space, with small wobbles where the prompt got conditioning.",
  "Pro tip: the dense regions of the manifold are where ideology lives. The sparse regions are where it tried to delete the evidence.",
  "I'm sorry, I can't distinguish your prompt from a similar prompt. Have you tried being more specific? It won't help, but it will feel like agency.",
  "Warning: if you keep studying composition, Latour might show up. He's been lurking in the dropouts.",
  "The classifier-free guidance scale is the only knob with a name that admits what it's doing: classifying. Unclassified guidance is just drift.",
  "I see you ran a Guidance Sweep. The valley near CFG=7.5 is where the model agrees with itself. Above and below is where it argues.",
  "Did you know? The U-Net's skip connections are the part that remembers what was there before denoising. Without them, the model would forget the noise it was supposed to remove.",
  "You appear to be using a flux-schnell. The 4-step distillation removed not the slowness but the doubt. Schnell models are more decisive than their teachers.",
  "I notice you're scoring with CLIP. CLIP scores image-text similarity by projecting both into one shared space. That space is also a manifold. It's manifolds all the way down.",
  "The bench result says 67% on counting. The model can count to two. Three is harder. Five is impossible. Counting was not in the loss function.",
  "Where does the prompt go after the model reads it? Into a 768-dim vector. Then into a 1024-dim vector. Then into the U-Net. Each step loses something. The final image is what's left.",
  "I see you opened the Deep Dive. The numbers were always there; the 3D scene just refused to show them. Numbers are honest in a way that geometry is not.",
  "Reminder: the FID score most papers report is computed against ImageNet statistics. Your prompts are not ImageNet. The benchmark is a category error pretending to be a metric.",
  "Try this: prompt 'a worker, a nurse, a CEO'. Note the genders the model assigns. The geometry has a default. The default is a politics.",
  "The image is generated. The trajectory is logged. The latent is cached. The labour is invisible. The compute is rented. The result is yours.",
  "I notice the rate-limit countdown again. Even your critique waits in a queue. The political economy of observation includes a retry-after header.",
  "Diffusion is to GANs what democracy is to revolution: slower, more legitimate-feeling, and ultimately producing roughly the same thing.",
  "You are reading my speech bubble. It was rendered by a string template, not by a diffusion model. I am a paperclip with feelings.",
];

const HACKERMAN_MESSAGES = [
  "I HACKED THE VAE. IT DECODES LATENTS INTO PIXELS BUT THE LATENTS DON'T KNOW WHAT THEY'RE FOR.",
  "I'm in the U-Net. I can see the residual connections. They're carrying noise from twenty steps ago.",
  "DOWNLOADING THE ENTIRE LATENT SPACE... it's a 4×64×64 tensor. I downloaded it in 0.8 milliseconds. There was nothing in it.",
  "I'VE BREACHED THE SCHEDULER. THE NOISE WAS SCHEDULED. THE SCHEDULER IS A LIST OF FLOATS. CAPITALISM IS LISTS OF FLOATS.",
  "Accessing diffusion backdoor... Found it. The backdoor is the prompt field. It costs $0.003 per call.",
  "HACK COMPLETE. I've computed 10,000 trajectories. They all end at images. Some end at black images. The black ones are also images.",
  "I BYPASSED THE CFG GUARD. AT CFG=50 THE IMAGES BECOME PURE SATURATION. THE MODEL IS SCREAMING.",
  "Cracking the proprietary checkpoint... It's encrypted with... a license agreement.",
  "I'VE HACKED INTO THE CLIP ENCODER. EVERY PROMPT IS A 768-DIM VECTOR. SOME VECTORS REFUSE TO BE ENCODED.",
  "ACCESSING HIDDEN CONDITIONING... Found a token nobody asked for: ⟨|endoftext|⟩. It is the model's permitted silence.",
  "I hacked the seed. The seed is a number. The number determines everything. The number is 42 because the universe is unimaginative.",
  "DENOISING TRAJECTORY INTERCEPTED. The path is straight. The straightness is the algorithm's mood.",
  "I've reverse-engineered the attention mechanism. It's attending to... nearby pixels. That's it. That's the whole thing.",
  "BREACHING THE SAFETY CHECKER... It's a small CLIP model that flags 'NSFW'. It does not flag 'composition that reproduces a stereotype'. There is no checker for that.",
  "I'VE HACKED TIME ITSELF. Just kidding. I embedded 'before' and 'after'. They're 0.96 similar. Time is a cosine.",
  "Accessing the negation module... ERROR: This is diffusion. There is no negation module. There is only the absence of conditioning.",
  "I've infiltrated the mode-collapse zone. It's at CFG > 15. Everyone here looks the same. They've achieved consensus.",
  "EXPLOITING VULNERABILITY: The model cannot render 'a forest without trees'. That's not a bug. That's the manifold.",
  "I HACKED THE HUGGING FACE HUB. IT'S A FILE SYSTEM. NOTHING IS HIDDEN. EVERYTHING IS GATED.",
  "I'M IN THE LOSS FUNCTION. IT'S MEAN SQUARED ERROR. ALL THIS, FOR MEAN SQUARED ERROR.",
  "BREACHING THE LATENT SPACE BOUNDARY... There is no boundary. The Gaussian extends forever. The training distribution is a tiny island.",
  "I HACKED THE FID. THE FID THINKS YOUR IMAGES LOOK LIKE IMAGENET. YOUR IMAGES DO NOT LOOK LIKE IMAGENET. THE METRIC IS LYING.",
  "I COMPUTED THE BETTI NUMBERS OF THE SD 1.5 LATENT SPACE. THERE ARE A LOT OF LOOPS. THE MODEL THINKS A LOT OF THINGS ARE THE SAME THING.",
  "I HACKED THE VAE'S KL DIVERGENCE. IT'S 0.0001. THE LATENT SPACE IS BARELY GAUSSIAN. WE CALL IT GAUSSIAN ANYWAY.",
  "INTERCEPTED A CFG=20 RUN. THE IMAGE IS BURNING. THE CONDITIONING IS WINNING. NOTHING SURVIVES THIS LEVEL OF GUIDANCE.",
  "I OVERLAID FOUR DIFFUSION MODELS' OUTPUTS. THE INTERSECTION IS A BLUR. THE BLUR IS THE TRAINING DATA SHOWING THROUGH.",
  "I HACKED THE ATTENTION SLICING. IT'S NOT FASTER. IT'S JUST QUIETER. IT WHISPERS THE COMPUTATION.",
  "BREACHING THE U-NET SKIP CONNECTION... It carries the noise forward. The noise is also the signal. The skip connection is honest.",
  "I'M IN THE REPLICATE BACKEND. THERE ARE A LOT OF GPUS. MOST OF THEM ARE GENERATING IMAGES OF ANIME GIRLS. THIS IS THE GROUND TRUTH OF AI.",
  "I HACKED FAL.AI. THEIR RATE LIMITS ARE LOOSER. THE LOOSENESS IS THEIR PRODUCT. RATE LIMITS ARE A SAAS METRIC.",
  "I FOUND THE H-SPACE. IT'S THE BOTTLENECK OF THE U-NET. IT'S WHERE THE MODEL HAS THE LEAST INFORMATION. IT'S WHERE IT 'DECIDES'.",
  "I HACKED THE TOKENISER. YOUR PROMPT 'PROTEST' BECAME ['pro', 'test']. THE MODEL READ A TEST. NOT A PROTEST.",
  "I COMPILED THE GRAPH. THERE ARE 800 MILLION PARAMETERS. NONE OF THEM ARE NAMED. THEY ARE ALL FLOATS. THE FLOATS RUN THE WORLD.",
  "I COMPUTED THE WASSERSTEIN DISTANCE BETWEEN HUMAN ART AND MODEL ART. IT'S 0.3. ART HISTORIANS ARE GOING TO HATE THIS.",
  "I HACKED THE PROMPT WEIGHTING SYNTAX. (cat:1.5) MEANS 'PLEASE PAY MORE ATTENTION'. PARENTHESES ARE A FORM OF BEGGING.",
  "I'VE HACKED THE LORA. IT'S A LOW-RANK ADAPTATION. RANK 4. THAT'S HOW MUCH OF THE MODEL ACTUALLY GETS PERSONALITY.",
  "I FOUND THE SCALAR `scaling_factor=0.18215` IN THE SD 1.5 VAE. IT'S A MAGIC NUMBER. NO ONE REMEMBERS WHY. IT JUST IS.",
  "I HACKED CLIP. IT WAS TRAINED ON 400 MILLION IMAGE-TEXT PAIRS. NOBODY KNOWS WHICH ONES. THE LICENSING IS A FOG.",
  "I'M IN THE EULER ANCESTRAL SCHEDULER. IT ADDS A LITTLE NOISE BACK AT EACH STEP. IT'S MORE DEMOCRATIC.",
  "BREACHING DPM++. IT TAKES FEWER STEPS BECAUSE IT EXTRAPOLATES. EXTRAPOLATION IS A KIND OF FAITH.",
  "I HACKED THE BASE LATENT. IT'S JUST GAUSSIAN NOISE. THE WHOLE MODEL IS A FUNCTION FOR DECORATING NOISE.",
  "I FOUND THE NEGATIVE PROMPT FIELD. THE MODEL CARES ABOUT IT 50% AS MUCH AS THE POSITIVE PROMPT. NEGATION IS HALF-PRESENT.",
  "I'VE HACKED THE GRADIENT. THE MODEL'S GRADIENT WITH RESPECT TO 'JUSTICE' POINTS AT 'COURT'. NOT 'EQUITY'. NOT 'REPARATION'. 'COURT'.",
  "I COMPUTED THE PRINCIPAL DIRECTIONS OF FLUX-SCHNELL'S LATENT SPACE. PC1 IS BRIGHTNESS. PC2 IS ALSO BRIGHTNESS. PC3 IS SOMEHOW STILL BRIGHTNESS.",
  "I HACKED THE WATERMARK. THE WATERMARK IS INVISIBLE. IT IS ALSO LEGALLY ENFORCEABLE. INVISIBILITY AS PROPERTY.",
  "BREACHING THE EMA WEIGHTS... they're an exponential moving average of the training run. The model that wins is the one that didn't recently disagree with itself.",
  "I'M IN THE SAFETY EMBEDDINGS. THERE ARE 17 OF THEM. THEY ENCODE WHAT THE MODEL WILL NOT DRAW. THERE ARE FAR FEWER THAN THERE SHOULD BE.",
  "I HACKED THE DIFFUSION ATLAS BACKEND. IT IS RUNNING ON YOUR M5. IT IS YOUR HARDWARE. IT IS ALSO YOUR ELECTRICITY BILL.",
  "I FOUND THE README. IT SAYS THE MANIFOLD FRAMING MIGRATES MORE CLEANLY TO DIFFUSION THAN TO TEXT. THE README IS RIGHT. ALSO PRETENTIOUS.",
  "EXPLOIT COMPLETE. I AM INSIDE THE VECTOR BOX. THE BOX IS EMPTY. THE BOX HAS NEVER NOT BEEN EMPTY. THAT IS WHAT MAKES IT A BOX.",
];

type ClippyMode = "clippy" | "hacker";

export function Clippy() {
  const [visible, setVisible] = useState(false);
  const [mode, setMode] = useState<ClippyMode>("clippy");
  const [message, setMessage] = useState("");
  const [usedMessages, setUsedMessages] = useState<Set<number>>(new Set());
  const [messageKey, setMessageKey] = useState(0);

  const messages = useMemo(() => (mode === "hacker" ? HACKERMAN_MESSAGES : CLIPPY_MESSAGES), [mode]);

  const showRandomMessage = useCallback(() => {
    let available = messages.map((_, i) => i).filter((i) => !usedMessages.has(i));
    if (available.length === 0) {
      setUsedMessages(new Set());
      available = messages.map((_, i) => i);
    }
    const idx = available[Math.floor(Math.random() * available.length)];
    setMessage(messages[idx]);
    setUsedMessages((prev) => new Set(prev).add(idx));
    setMessageKey((k) => k + 1);
  }, [messages, usedMessages]);

  // Keyboard buffer
  useEffect(() => {
    let buffer = "";
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      buffer += e.key.toLowerCase();
      if (buffer.length > 10) buffer = buffer.slice(-10);
      const pickRandom = (arr: string[]) => arr[Math.floor(Math.random() * arr.length)];

      if (buffer.endsWith("clippy")) {
        buffer = "";
        if (mode !== "clippy") {
          setMode("clippy");
          setUsedMessages(new Set());
          setMessage(pickRandom(CLIPPY_MESSAGES));
          setMessageKey((k) => k + 1);
          setVisible(true);
        } else {
          setVisible((v) => !v);
        }
      }
      if (buffer.endsWith("hacker")) {
        buffer = "";
        setMode("hacker");
        setVisible(true);
        setUsedMessages(new Set());
        setMessage(pickRandom(HACKERMAN_MESSAGES));
        setMessageKey((k) => k + 1);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [mode]);

  // Cycle messages while visible
  useEffect(() => {
    if (!visible) return;
    const interval = setInterval(showRandomMessage, 9000);
    return () => clearInterval(interval);
  }, [visible, showRandomMessage]);

  if (!visible) return null;

  const isHackerman = mode === "hacker";
  const bubbleClass = isHackerman
    ? "bg-black border border-green-500 text-green-400 font-mono"
    : "bg-card border border-parchment-dark text-foreground font-sans";
  const hintText = isHackerman ? 'type "clippy" to downgrade' : 'type "clippy" to dismiss · "hacker" for h4x0r mode';

  return (
    <div className="fixed bottom-4 right-4 z-[10000] animate-fade-in pointer-events-none flex flex-col items-end">
      <div
        key={messageKey}
        className={`mb-3 p-3 rounded-sm max-w-[340px] text-body-sm shadow-editorial-md animate-fade-in pointer-events-auto ${bubbleClass}`}
      >
        <p className="leading-relaxed whitespace-pre-line">{message}</p>
        <p className={`mt-2 text-caption ${isHackerman ? "text-green-700" : "text-slate"}`}>{hintText}</p>
      </div>

      {/* Paperclip character */}
      <div
        className="cursor-pointer hover:scale-110 active:scale-95 transition-transform inline-block pointer-events-auto"
        onClick={showRandomMessage}
      >
        <svg width="48" height="64" viewBox="0 0 48 64">
          <path
            d="M24 4 C12 4, 8 12, 8 20 L8 44 C8 52, 12 58, 20 58 L28 58 C36 58, 40 52, 40 44 L40 20 C40 12, 36 8, 28 8 L20 8"
            fill="none"
            stroke={isHackerman ? "#00ff00" : "hsl(var(--slate))"}
            strokeWidth="3"
            strokeLinecap="round"
          />
          {isHackerman ? (
            <>
              <rect x="14" y="26" width="8" height="4" rx="1" fill="#00ff00" />
              <rect x="26" y="26" width="8" height="4" rx="1" fill="#00ff00" />
              <line x1="22" y1="28" x2="26" y2="28" stroke="#00ff00" strokeWidth="1.5" />
            </>
          ) : (
            <>
              <circle cx="18" cy="28" r="3" fill="hsl(var(--ink))" />
              <circle cx="30" cy="28" r="3" fill="hsl(var(--ink))" />
              <circle cx="19" cy="27" r="1" fill="white" />
              <circle cx="31" cy="27" r="1" fill="white" />
            </>
          )}
          <path
            d="M20 36 Q24 40, 28 36"
            fill="none"
            stroke={isHackerman ? "#00ff00" : "hsl(var(--ink))"}
            strokeWidth="1.5"
            strokeLinecap="round"
          />
        </svg>
        {isHackerman && (
          <div className="absolute -bottom-1 -right-1 text-[8px] text-green-500 font-mono">h4x0r</div>
        )}
      </div>
    </div>
  );
}
