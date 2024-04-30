import CodeSection from "./CodeSection.jsx";
import React from "react";
function CreateSection(props){
return(
    <>
    <CodeSection key={props.id} name={props.name} code={props.code}/>
    </>
);
}
export default CreateSection;